from itertools import groupby

from devito.ir.clusters import Cluster, Queue
from devito.ir.support import TILABLE
from devito.symbolics import xreplace_indices
from devito.tools import filter_ordered
from devito.types import Scalar

__all__ = ['Lift', 'fuse', 'scalarize', 'eliminate_arrays']


class Lift(Queue):

    """
    Remove invariant Dimensions from Clusters to avoid redundant computation.

    Notes
    -----
    This is analogous to the compiler transformation known as
    "loop-invariant code motion".
    """

    def callback(self, clusters, prefix):
        if not prefix:
            # No iteration space to be lifted from
            return clusters

        hope_invariant = {i.dim for i in prefix}

        lifted = []
        processed = []
        for n, c in enumerate(clusters):
            # Increments prevent lifting
            if c.has_increments:
                processed.append(c)
                continue

            # Is `c` a real candidate -- is there at least one invariant Dimension?
            if c.used_dimensions & hope_invariant:
                processed.append(c)
                continue

            impacted = set(processed) | set(clusters[n+1:])

            # None of the Functions appearing in a lifted Cluster can be written to
            if any(c.functions & set(i.scope.writes) for i in impacted):
                processed.append(c)
                continue

            # Scalars prevent lifting if they are read by another Cluster
            swrites = {f for f in c.scope.writes if f.is_Symbol}
            if any(swrites & set(i.scope.reads) for i in impacted):
                processed.append(c)
                continue

            # Contract iteration and data spaces for the lifted Cluster
            key = lambda d: d not in hope_invariant
            ispace = c.ispace.project(key).reset()
            dspace = c.dspace.project(key).reset()

            # Some properties need to be dropped
            properties = {d: v for d, v in c.properties.items() if key(d)}
            properties = {d: v - {TILABLE} for d, v in properties.items()}

            lifted.append(c.rebuild(ispace=ispace, dspace=dspace, properties=properties))

        return lifted + processed


def fuse(clusters):
    """
    Fuse sub-sequences of Clusters with compatible IterationSpace.
    """
    key = lambda c: (set(c.itintervals), c.guards)

    processed = []
    for k, g in groupby(clusters, key=key):
        maybe_fusible = list(g)

        if len(maybe_fusible) == 1:
            processed.extend(maybe_fusible)
        else:
            try:
                # Perform fusion
                fused = Cluster.from_clusters(*maybe_fusible)
                processed.append(fused)
            except ValueError:
                # We end up here if, for example, some Clusters have same
                # iteration Dimensions but different (partial) orderings
                processed.extend(maybe_fusible)

    return processed


def scalarize(clusters, template):
    """
    Turn local "isolated" Arrays, that is Arrays appearing only in one Cluster,
    into Scalars.
    """
    processed = []
    for c in clusters:
        # Get any Arrays appearing only in `c`
        impacted = set(clusters) - {c}
        arrays = {i for i in c.scope.writes if i.is_Array}
        arrays -= set().union(*[i.scope.reads for i in impacted])

        # Turn them into scalars
        #
        # r[x,y,z] = g(b[x,y,z])                 t0 = g(b[x,y,z])
        # ... = r[x,y,z] + r[x,y,z+1]`  ---->    t1 = g(b[x,y,z+1])
        #                                        ... = t0 + t1
        mapper = {}
        exprs = []
        for e in c.exprs:
            f = e.lhs.function
            if f in arrays:
                for i in filter_ordered(i.indexed for i in c.scope[f]):
                    mapper[i] = Scalar(name=template(), dtype=f.dtype)

                    assert len(f.indices) == len(e.lhs.indices) == len(i.indices)
                    shifting = {idx: idx + (o2 - o1) for idx, o1, o2 in
                                zip(f.indices, e.lhs.indices, i.indices)}

                    handle = e.func(mapper[i], e.rhs.xreplace(mapper))
                    handle = xreplace_indices(handle, shifting)
                    exprs.append(handle)
            else:
                exprs.append(e.func(e.lhs, e.rhs.xreplace(mapper)))

        processed.append(c.rebuild(exprs))

    return processed


def eliminate_arrays(clusters, template):
    """
    Eliminate redundant expressions stored in Arrays.
    """
    mapper = {}
    processed = []
    for c in clusters:
        if not c.is_dense:
            processed.append(c)
            continue

        # Search for any redundant RHSs
        seen = {}
        for e in c.exprs:
            f = e.lhs.function
            if not f.is_Array:
                continue
            v = seen.get(e.rhs)
            if v is not None:
                # Found a redundant RHS
                mapper[f] = v
            else:
                seen[e.rhs] = f

        if not mapper:
            # Do not waste time
            processed.append(c)
            continue

        # Replace redundancies
        subs = {}
        for f, v in mapper.items():
            for i in filter_ordered(i.indexed for i in c.scope[f]):
                subs[i] = v[f.indices]
        exprs = []
        for e in c.exprs:
            if e.lhs.function in mapper:
                # Drop the write
                continue
            exprs.append(e.xreplace(subs))

        processed.append(c.rebuild(exprs))

    return processed
