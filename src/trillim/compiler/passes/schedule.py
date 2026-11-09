from __future__ import annotations

from ...tdl.core.core_dataclasses import _Region
from ...tdl.core.core_pydantic_models import (
    _BroadcastArgs,
    _ConstantArgs,
    _ConstructArgs,
    _InputArgs,
    _PermuteArgs,
    _ReshapeArgs,
)
from ...tdl.core.graph import Graph
from ...tdl.core.operator import _Operator
from ...tdl.core.tensor import Tensor
from ..artifacts.expressions import _byte_size, _parameter_references, _references
from ..artifacts.parameters import _requirements
from ..compiler_dataclasses import _Inference
from .passes_dataclasses import _Selection
from .walk import _ordered_operators, _predecessors


def _storage(tensor: Tensor, selection: _Selection) -> Tensor:
    while (producer := tensor._producer) is not None:
        if producer in selection.covered:
            break
        args = producer.args
        if isinstance(args, _ConstructArgs):
            tensor = producer.regions[0].outputs[producer.outputs.index(tensor)]
        elif isinstance(args, (_PermuteArgs, _BroadcastArgs, _ReshapeArgs)):
            tensor = args.x
        else:
            break
    return tensor


def _size(tensor: Tensor, inference: _Inference) -> int:
    if tensor._producer is not None and isinstance(
        tensor._producer.args, (_ConstantArgs, _InputArgs)
    ):
        return 0
    info = inference.tensors[tensor]
    if (
        info.dtype is None
        or info.maximum_shape is None
        or any(item is None for item in info.maximum_shape)
    ):
        return 0
    return _byte_size(
        info.dtype, tuple(item for item in info.maximum_shape if item is not None)
    )


def _schedule(graph: Graph, inference: _Inference, selection: _Selection) -> None:
    """Greedily order selected calls while preserving data and effect dependencies."""
    if graph._region is None:
        raise ValueError("graph is not complete")
    owners = {
        node: call.anchor for call in selection.calls.values() for node in call.covered
    }
    metadata: dict[_Operator, set[Tensor]] = {}
    for constraint in inference.constraints:
        if constraint.operator is not None:
            metadata.setdefault(constraint.operator, set()).update(
                _references(constraint.left) | _references(constraint.right)
            )
    requirements = {
        operator: _requirements(call.kernel.parameter_schema, call.parameters)
        for operator, call in selection.calls.items()
    }

    def region_order(region: _Region) -> None:
        for operator in region.operators:
            if operator not in selection.covered:
                for body in operator.regions:
                    region_order(body)
        local = set(region.operators)
        indices = {operator: index for index, operator in enumerate(region.operators)}
        groups: dict[_Operator, list[_Operator]] = {}
        for operator in _ordered_operators(region):
            anchor = owners.get(operator, operator)
            if anchor not in local:
                raise ValueError("kernel coverage crosses a scheduling region")
            groups.setdefault(anchor, []).append(operator)
        dependencies: dict[_Operator, set[_Operator]] = {}
        inputs: dict[_Operator, set[Tensor]] = {}
        outputs: dict[_Operator, set[Tensor]] = {}
        for operator, members in groups.items():
            call = selection.calls.get(operator)
            nodes = call.covered if call else members
            reads = set(call.bindings.values() if call else operator.inputs)
            if call is not None:
                for parameter in call.parameters.values():
                    reads.update(_parameter_references(parameter))
            for node in nodes:
                reads.update(metadata.get(node, ()))
            reads.difference_update(operator.outputs)
            required = {
                owners.get(predecessor, predecessor)
                for node in nodes
                for predecessor in _predecessors(node)
                if predecessor in local
            }
            required.update(
                owners.get(tensor._producer, tensor._producer)
                for tensor in reads
                if tensor._producer in local
            )
            required.discard(operator)
            dependencies[operator] = required
            inputs[operator] = {_storage(tensor, selection) for tensor in reads}
            outputs[operator] = {
                _storage(tensor, selection)
                for tensor in (call.outputs if call else operator.outputs)
            }
        followers: dict[_Operator, list[_Operator]] = {node: [] for node in groups}
        uses: dict[Tensor, int] = {}
        for operator in groups:
            for dependency in dependencies[operator]:
                followers[dependency].append(operator)
            for tensor in inputs[operator]:
                uses[tensor] = uses.get(tensor, 0) + 1
        for tensor in region.outputs:
            root = _storage(tensor, selection)
            uses[root] = uses.get(root, 0) + 1
        sizes = {
            tensor: _size(tensor, inference)
            for operator in groups
            for tensor in inputs[operator] | outputs[operator]
        }
        for operator in groups:
            call = selection.calls.get(operator)
            if call is not None:
                for tensor, size in zip(
                    call.outputs, requirements[operator].outputs, strict=True
                ):
                    root = _storage(tensor, selection)
                    sizes[root] = max(sizes[root], size)

        def score(operator: _Operator) -> tuple[int, int]:
            released = sum(
                sizes[tensor] for tensor in inputs[operator] if uses[tensor] == 1
            )
            allocated = sum(
                sizes[tensor] for tensor in outputs[operator] - inputs[operator]
            )
            scratch = requirements[operator].scratch if operator in requirements else 0
            return allocated + scratch - released, indices[operator]

        ready = [operator for operator in groups if not dependencies[operator]]
        result: list[_Operator] = []
        while ready:
            operator = min(ready, key=score)
            ready.remove(operator)
            # Keep absorbed nodes for graph traversal; execution skips them.
            result.extend(groups[operator])
            for tensor in inputs[operator]:
                uses[tensor] -= 1
            for follower in followers[operator]:
                dependencies[follower].remove(operator)
                if not dependencies[follower]:
                    ready.append(follower)
        if len(result) != len(local):
            raise ValueError("cannot schedule a cyclic kernel plan")
        region.operators = tuple(result)

    region_order(graph._region)
