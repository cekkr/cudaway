"""CUDA→ROCm driver API + conversion placeholder generator."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Dict, Iterable, List, Tuple


ParamDirectionLiteral = str
ConversionKindLiteral = str


@dataclass
class ParameterSpec:
    name: str
    direction: ParamDirectionLiteral
    cuda_type: str
    rocm_type: str
    notes: str


@dataclass
class FunctionSpec:
    cuda_name: str
    cuda_return_type: str
    rocm_name: str
    rocm_return_type: str
    notes: str
    parameters: List[ParameterSpec]


@dataclass
class ConversionRecord:
    kind: ConversionKindLiteral
    source_type: str
    target_type: str
    contexts: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add_context(self, context: str) -> None:
        if context and context not in self.contexts:
            self.contexts.append(context)

    def add_note(self, note: str) -> None:
        note = note.strip()
        if note and note not in self.notes:
            self.notes.append(note)

    def description(self) -> str:
        parts: List[str] = []
        if self.contexts:
            parts.append("Contexts: " + ", ".join(sorted(self.contexts)))
        if self.notes:
            parts.append("Notes: " + " | ".join(self.notes))
        return " | ".join(parts) if parts else "Auto-generated placeholder."


def load_spec(path: Path) -> Tuple[List[FunctionSpec], List[Dict[str, str]], Dict[str, str]]:
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    metadata = payload.get("metadata", {})
    functions: List[FunctionSpec] = []
    for raw_fn in payload.get("functions", []):
        params = [
            ParameterSpec(
                name=param["name"],
                direction=param["direction"],
                cuda_type=param["cuda_type"],
                rocm_type=param["rocm_type"],
                notes=param.get("notes", ""),
            )
            for param in raw_fn.get("parameters", [])
        ]
        functions.append(
            FunctionSpec(
                cuda_name=raw_fn["cuda_name"],
                cuda_return_type=raw_fn["cuda_return_type"],
                rocm_name=raw_fn["rocm_name"],
                rocm_return_type=raw_fn["rocm_return_type"],
                notes=raw_fn.get("notes", ""),
                parameters=params,
            )
        )
    conversions = payload.get("conversions", [])
    return functions, conversions, metadata


def escape_cpp(value: str) -> str:
    return (
        value.replace("\\", "\\\\")
        .replace("\"", r"\"")
        .replace("\n", "\\n")
    )


def sanitize_type_identifier(type_name: str) -> str:
    sanitized = type_name.strip()
    replacements = {
        "::": "_",
        "*": "_ptr",
        "&": "_ref",
        "<": "_",
        ">": "_",
        " ": "_",
        "[": "_arr_",
        "]": "_",
    }
    for needle, replacement in replacements.items():
        sanitized = sanitized.replace(needle, replacement)
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", sanitized)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "value"


def param_direction_enum(direction: str) -> str:
    mapping = {
        "in": "ParamDirection::kIn",
        "out": "ParamDirection::kOut",
        "inout": "ParamDirection::kInOut",
    }
    if direction not in mapping:
        raise ValueError(f"Unsupported parameter direction '{direction}'")
    return mapping[direction]


def conversion_kind_enum(kind: str) -> str:
    mapping = {
        "input": "ConversionKind::kInput",
        "output": "ConversionKind::kOutput",
        "return": "ConversionKind::kReturn",
    }
    if kind not in mapping:
        raise ValueError(f"Unsupported conversion kind '{kind}'")
    return mapping[kind]


def register_conversion(
    conversions: Dict[Tuple[str, str, str], ConversionRecord],
    kind: ConversionKindLiteral,
    source_type: str,
    target_type: str,
    *,
    context: str = "",
    note: str = "",
) -> None:
    key = (kind, source_type, target_type)
    record = conversions.get(key)
    if record is None:
        record = ConversionRecord(kind=kind, source_type=source_type, target_type=target_type)
        conversions[key] = record
    record.add_context(context)
    record.add_note(note)


def collect_conversions(
    functions: Iterable[FunctionSpec],
    custom_conversions: Iterable[Dict[str, str]],
) -> List[ConversionRecord]:
    conversions: Dict[Tuple[str, str, str], ConversionRecord] = {}
    for fn in functions:
        return_context = f"{fn.cuda_name}::return"
        register_conversion(
            conversions,
            "return",
            fn.rocm_return_type,
            fn.cuda_return_type,
            context=return_context,
            note="Return status requires translation.",
        )
        for param in fn.parameters:
            context = f"{fn.cuda_name}::{param.name}"
            note = param.notes
            direction = param.direction.lower()
            if direction not in {"in", "out", "inout"}:
                raise ValueError(f"Invalid direction '{param.direction}' for {context}")
            if direction in {"in", "inout"}:
                register_conversion(
                    conversions,
                    "input",
                    param.cuda_type,
                    param.rocm_type,
                    context=context,
                    note=note,
                )
            if direction in {"out", "inout"}:
                register_conversion(
                    conversions,
                    "output",
                    param.rocm_type,
                    param.cuda_type,
                    context=context,
                    note=note,
                )
    for entry in custom_conversions:
        direction = entry["direction"].lower()
        name = entry.get("name", "custom-conversion")
        context = f"custom::{name}"
        note = entry.get("notes", "")
        cuda_type = entry["cuda_type"]
        rocm_type = entry["rocm_type"]
        if direction == "in":
            register_conversion(
                conversions, "input", cuda_type, rocm_type, context=context, note=note
            )
        elif direction == "out":
            register_conversion(
                conversions, "output", rocm_type, cuda_type, context=context, note=note
            )
        elif direction == "return":
            register_conversion(
                conversions, "return", rocm_type, cuda_type, context=context, note=note
            )
        elif direction == "bidirectional":
            register_conversion(
                conversions, "input", cuda_type, rocm_type, context=context, note=note
            )
            register_conversion(
                conversions, "output", rocm_type, cuda_type, context=context, note=note
            )
        else:
            raise ValueError(f"Unsupported custom conversion direction '{direction}'")
    return [
        conversions[key]
        for key in sorted(
            conversions.keys(),
            key=lambda entry: (entry[0], entry[1], entry[2]),
        )
    ]


def render_parameter_storage(parameters: List[ParameterSpec]) -> Tuple[str, int]:
    lines = [
        f"inline constexpr std::array<ParameterMapping, {len(parameters)}> kParameterStorage{{",
    ]
    for param in parameters:
        lines.append(
            "    ParameterMapping{"
            f"\"{param.name}\", {param_direction_enum(param.direction.lower())}, "
            f"\"{param.cuda_type}\", \"{param.rocm_type}\", "
            f"\"{escape_cpp(param.notes)}\""
            "},"
        )
    lines.append("};")
    return "\n".join(lines), len(parameters)


def render_function_table(functions: List[FunctionSpec], parameters: List[ParameterSpec]) -> str:
    offset = 0
    lines = [
        f"inline constexpr std::array<ApiEntry, {len(functions)}> kDriverApiTable{{",
    ]
    for fn in functions:
        param_count = len(fn.parameters)
        lines.append(
            "    ApiEntry{"
            f"\"{fn.cuda_name}\", \"{fn.cuda_return_type}\", "
            f"\"{fn.rocm_name}\", \"{fn.rocm_return_type}\", "
            f"\"{escape_cpp(fn.notes)}\", "
            f"{offset}, {param_count}"
            "},"
        )
        offset += param_count
    lines.append("};")
    return "\n".join(lines)


def render_conversion_metadata(conversions: List[ConversionRecord]) -> str:
    lines = [
        f"inline constexpr std::array<ConversionPlaceholder, {len(conversions)}> "
        "kConversionPlaceholders{",
    ]
    for conv in conversions:
        lines.append(
            "    ConversionPlaceholder{"
            f"{conversion_kind_enum(conv.kind)}, "
            f"\"{conv.source_type}\", "
            f"\"{conv.target_type}\", "
            f"\"{escape_cpp(conv.description())}\""
            "},"
        )
    lines.append("};")
    return "\n".join(lines)


def render_conversion_functions(conversions: List[ConversionRecord]) -> Tuple[str, str]:
    fn_lines: List[str] = []
    macro_lines: List[str] = []
    for conv in conversions:
        source_id = sanitize_type_identifier(conv.source_type)
        target_id = sanitize_type_identifier(conv.target_type)
        context_comment = f"// {conv.description()}"
        if conv.kind == "input":
            fn_lines.extend(
                [
                    f"inline {conv.target_type} ConvertInput_{source_id}_to_{target_id}("
                    f"{conv.source_type} value, std::string_view functionName, "
                    "std::string_view fieldName) {",
                    "    (void)functionName;",
                    "    (void)fieldName;",
                    f"    {context_comment}",
                    f"    return static_cast<{conv.target_type}>(value);",
                    "}",
                    "",
                ]
            )
            macro_lines.append(
                "#define CUDAWAY_CONVERT_IN_"
                f"{source_id.upper()}_TO_{target_id.upper()}(functionName, fieldName, value) "
                f"::cudaway::host::generated::ConvertInput_{source_id}_to_{target_id}("
                "(value), (functionName), (fieldName))"
            )
        elif conv.kind == "output":
            fn_lines.extend(
                [
                    f"inline void ConvertOutput_{target_id}_from_{source_id}("
                    f"const {conv.source_type}& sourceValue, {conv.target_type}& destinationValue, "
                    "std::string_view functionName, std::string_view fieldName) {",
                    "    (void)functionName;",
                    "    (void)fieldName;",
                    "    (void)sourceValue;",
                    "    (void)destinationValue;",
                    f"    {context_comment}",
                    "}",
                    "",
                ]
            )
            macro_lines.append(
                "#define CUDAWAY_CONVERT_OUT_"
                f"{target_id.upper()}_FROM_{source_id.upper()}(functionName, fieldName, sourceValue, destinationValue) "
                f"::cudaway::host::generated::ConvertOutput_{target_id}_from_{source_id}("
                "(sourceValue), (destinationValue), (functionName), (fieldName))"
            )
        elif conv.kind == "return":
            fn_lines.extend(
                [
                    f"inline {conv.target_type} ConvertReturn_{target_id}_from_{source_id}("
                    f"{conv.source_type} value, std::string_view functionName) {{",
                    "    (void)functionName;",
                    f"    {context_comment}",
                    f"    return static_cast<{conv.target_type}>(value);",
                    "}",
                    "",
                ]
            )
            macro_lines.append(
                "#define CUDAWAY_CONVERT_RETURN_"
                f"{target_id.upper()}_FROM_{source_id.upper()}(functionName, value) "
                f"::cudaway::host::generated::ConvertReturn_{target_id}_from_{source_id}("
                "(value), (functionName))"
            )
        else:
            raise ValueError(f"Unsupported conversion kind '{conv.kind}'")
    return "\n".join(fn_lines), "\n".join(macro_lines)


def render_header(
    functions: List[FunctionSpec],
    conversions: List[ConversionRecord],
    spec_path: Path,
) -> str:
    parameters: List[ParameterSpec] = [
        param for fn in functions for param in fn.parameters
    ]
    parameter_block, _ = render_parameter_storage(parameters)
    function_block = render_function_table(functions, parameters)
    conversion_block = render_conversion_metadata(conversions)
    conversion_functions, macro_block = render_conversion_functions(conversions)
    generated_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "// Generated by tools/python/cuda_rocm_api_generator.py",
        f"// Spec: {spec_path}",
        f"// Generated at: {generated_at}",
        "#pragma once",
        "",
        "#include <array>",
        "#include <cstddef>",
        "#include <cstdint>",
        "#include <span>",
        "#include <string_view>",
        "",
        "#include \"host/CudaDriverTypes.hpp\"",
        "",
        "namespace cudaway::host::generated {",
        "",
        "#ifndef CUDAWAY_USE_NATIVE_CUDA_TYPES",
        "struct __nv_bfloat16 {",
        "    std::uint16_t storage{};",
        "};",
        "#endif  // CUDAWAY_USE_NATIVE_CUDA_TYPES",
        "",
        "#ifndef CUDAWAY_USE_NATIVE_HIP_TYPES",
        "using hipError_t = int;",
        "using hipDevice_t = int;",
        "using hipCtx_t = std::uintptr_t;",
        "using hipModule_t = std::uintptr_t;",
        "using hipFunction_t = std::uintptr_t;",
        "using hipStream_t = std::uintptr_t;",
        "struct hip_bfloat16 {",
        "    std::uint16_t storage{};",
        "};",
        "#endif  // CUDAWAY_USE_NATIVE_HIP_TYPES",
        "",
        "enum class ParamDirection {",
        "    kIn,",
        "    kOut,",
        "    kInOut,",
        "};",
        "",
        "enum class ConversionKind {",
        "    kInput,",
        "    kOutput,",
        "    kReturn,",
        "};",
        "",
        "struct ParameterMapping {",
        "    std::string_view name;",
        "    ParamDirection direction;",
        "    std::string_view cudaType;",
        "    std::string_view rocmType;",
        "    std::string_view notes;",
        "};",
        "",
        "struct ApiEntry {",
        "    std::string_view cudaName;",
        "    std::string_view cudaReturnType;",
        "    std::string_view rocmName;",
        "    std::string_view rocmReturnType;",
        "    std::string_view notes;",
        "    std::size_t paramOffset;",
        "    std::size_t paramCount;",
        "};",
        "",
        "struct ConversionPlaceholder {",
        "    ConversionKind kind;",
        "    std::string_view sourceType;",
        "    std::string_view targetType;",
        "    std::string_view description;",
        "};",
        "",
        parameter_block,
        "",
        function_block,
        "",
        "inline constexpr std::span<const ParameterMapping> parameters_for(const ApiEntry& entry) {",
        "    return {kParameterStorage.data() + entry.paramOffset, entry.paramCount};",
        "}",
        "",
        conversion_block,
        "",
        conversion_functions,
        "",
        macro_block,
        "",
        "}  // namespace cudaway::host::generated",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(content)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        default="tools/data/cuda_rocm_driver_apis.json",
        type=Path,
        help="Path to the CUDA→ROCm API spec JSON.",
    )
    parser.add_argument(
        "--header-out",
        default="src/host/generated/CudaRocmApi.generated.hpp",
        type=Path,
        help="Path to the generated C++ header.",
    )
    args = parser.parse_args(argv)

    functions, extra_conversions, metadata = load_spec(args.spec)
    conversions = collect_conversions(functions, extra_conversions)
    header = render_header(functions, conversions, spec_path=args.spec)
    write_file(args.header_out, header)
    print(
        f"Generated {len(functions)} API entries, "
        f"{sum(len(fn.parameters) for fn in functions)} parameters, "
        f"and {len(conversions)} conversion placeholders."
    )
    print(f"Header -> {args.header_out}")
    if metadata:
        print(f"Spec metadata: {metadata.get('description', 'n/a')}")


if __name__ == "__main__":
    main()
