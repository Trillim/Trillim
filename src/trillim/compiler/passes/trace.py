from __future__ import annotations

from ...tdl.core import Graph
from ..compiler_pydantic_models import CompileRequest


def trace(request: CompileRequest) -> Graph:
    filename = str(request.tdl_path)
    namespace: dict[str, object] = {"__name__": "__tdl__", "__file__": filename}
    code = compile(request.tdl_path.read_bytes(), filename, "exec", dont_inherit=True)
    exec(code, namespace)  # noqa: S102 - TDL programs execute to construct their graphs.
    graph = namespace.get("graph")
    if not isinstance(graph, Graph):
        raise TypeError("TDL file must define 'graph' as a Graph")
    if graph.region is None:
        raise ValueError("TDL graph must be completed using its context manager")
    return graph
