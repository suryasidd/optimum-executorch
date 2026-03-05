# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Dict, Union

import torch
from tabulate import tabulate
from torch.export import ExportedProgram

from executorch.backends.openvino.partitioner import OpenvinoPartitioner
from executorch.exir.backend.backend_details import CompileSpec
from executorch.devtools.backend_debug import get_delegation_info
from executorch.exir import (
    EdgeCompileConfig,
    ExecutorchBackendConfig,
    ExecutorchProgram,
    to_edge_transform_and_lower,
)
from executorch.exir.passes import MemoryPlanningPass
from optimum.executorch.passes.remove_padding_idx_embedding_pass import RemovePaddingIdxEmbeddingPass

from ..integrations import (
    CausalLMExportableModule,
    MaskedLMExportableModule,
    MultiModalTextToTextExportableModule,
    Seq2SeqLMExportableModule,
    VisionEncoderExportableModule,
)
from ..recipe_registry import register_recipe


def _apply_openvino_quantization(
    exported_programs: Dict[str, ExportedProgram],
    mode: str,
    group_size: int = 128,
) -> Dict[str, ExportedProgram]:
    """
    Apply OpenVINO weight-only quantization to exported program.

    """
    try:
        from executorch.backends.openvino.quantizer import (
            OpenVINOQuantizer,
            QuantizationMode,
        )
    except:
        raise ImportError("OpenVINO quantizer not found. Please install the required dependencies.")
    quantizer_params = {}
    quantizer_params["mode"] = QuantizationMode.INT4WO_SYM
    quantizer_params["group_size"] = group_size
    quantizer_params["ratio"] = 1

    quantizer = OpenVINOQuantizer(**quantizer_params)

    from executorch.backends.openvino.quantizer import apply_nncf_data_aware_compression

    quantized_program = apply_nncf_data_aware_compression(exported_programs, quantizer, True, True)

    return quantized_program


@register_recipe("openvino")
def export_to_executorch_with_openvino(
    model: Union[
        CausalLMExportableModule,
        MaskedLMExportableModule,
        Seq2SeqLMExportableModule,
        MultiModalTextToTextExportableModule,
        VisionEncoderExportableModule,
    ],
    **kwargs,
):
    def _lower_to_executorch(
        exported_programs: Dict[str, ExportedProgram],
        metadata=None,
        device: str = "CPU",
        enable_memory_planning: bool = True,
    ) -> Dict[str, ExecutorchProgram]:
        backend_config_dict = {
            "extract_delegate_segments": True,
        }

        if enable_memory_planning:
            backend_config_dict["memory_planning_pass"] = MemoryPlanningPass(alloc_graph_input=False)

        logging.debug(f"\nExported program: {exported_programs}")
        logging.info(f"OpenVINO backend configuration - Device: {device}")

        # If just one exported program, the method name in the .pte for it should be "forward".
        if len(exported_programs) == 1:
            exported_programs = {"forward": next(iter(exported_programs.values()))}

        # Create OpenVINO partitioner with compile specs
        compile_specs = [
            CompileSpec("device", device.encode()),
        ]
        openvino_partitioner = OpenvinoPartitioner(
            compile_spec=compile_specs,
        )

        et_prog = to_edge_transform_and_lower(
            exported_programs,
            partitioner=[openvino_partitioner],
            compile_config=EdgeCompileConfig(
                _check_ir_validity=False,
                _skip_dim_order=True,
            ),
            constant_methods=metadata,
            transform_passes=[RemovePaddingIdxEmbeddingPass()],
        )
        et_prog = et_prog.to_executorch(
            config=ExecutorchBackendConfig(**backend_config_dict),
        )
        pte_name = "model"
        for method in et_prog.methods:
            logging.debug(f"---------------------- Method: {method} ----------------------")
            logging.debug(f"\nExecuTorch program for {pte_name}.pte: {et_prog.exported_program(method).graph_module}")
            delegation_info = get_delegation_info(et_prog.exported_program(method).graph_module)
            logging.debug(f"\nDelegation info Summary for {pte_name}.pte: {delegation_info.get_summary()}")
            logging.debug(
                f"\nDelegation info for {pte_name}.pte: {tabulate(delegation_info.get_operator_delegation_dataframe(), headers='keys', tablefmt='fancy_grid')}"
            )
        return {pte_name: et_prog}

    # Extract OpenVINO-specific parameters from kwargs
    device = kwargs.get("device", "CPU")

    enable_memory_planning = kwargs.get("enable_memory_planning", True)
    openvino_quantization = kwargs.get("openvino_quantization", None)
    openvino_group_size = kwargs.get("openvino_group_size", 128)

    exported_progs = model.export()

    # Apply OpenVINO PT2E quantization if requested (post-export, NNCF-backed)
    if openvino_quantization:
        exported_progs = _apply_openvino_quantization(
            exported_progs,
            mode=openvino_quantization,
            group_size=openvino_group_size,
        )

    if (
        model.config._attn_implementation == "custom_sdpa"
        or model.config._attn_implementation == "custom_sdpa_ring_kv_cache"
    ):
        # Sanity check to make sure the exported program contains the custom sdpa operator.
        if not any(
            node.op == "call_function" and "custom_sdpa" in str(node.target)
            for exported_program in exported_progs.values()
            for node in exported_program.graph_module.graph.nodes
        ):
            raise ValueError("'custom_sdpa' not found in the graph.")

    return _lower_to_executorch(
        exported_progs,
        model.metadata,
        device=device,
        enable_memory_planning=enable_memory_planning,
    )
