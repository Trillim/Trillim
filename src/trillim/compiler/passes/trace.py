from __future__ import annotations

from ...tdl.core import Graph
from ..compiler_pydantic_models import _CompileRequest


def _trace(request: _CompileRequest) -> Graph:
    filename = str(request.tdl_path)
    namespace: dict[str, object] = {"__name__": "__tdl__", "__file__": filename}
    try:
        code = compile(
            request.tdl_path.read_bytes(), filename, "exec", dont_inherit=True
        )
        exec(code, namespace)  # noqa: S102 - TDL programs execute to construct their graphs.
    except Exception as error:
        error.add_note(f"While tracing TDL file {filename!r}")
        raise
    graph = namespace.get("graph")
    if not isinstance(graph, Graph):
        raise TypeError(f"{filename}: TDL file must define 'graph' as a Graph")
    if graph._region is None:
        raise ValueError(
            f"{filename}: TDL graph must be completed using its context manager"
        )
    return graph
