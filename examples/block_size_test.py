from examples.seismic.acoustic.acoustic_example import acoustic_setup
from argparse import ArgumentParser

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_blocksizes(op, dle, grid, blockshape, level=0):
    blocksizes = {'%s0_blk%d_size' % (d, level): v
                  for d, v in zip(grid.dimensions, blockshape)}
    blocksizes = {k: v for k, v in blocksizes.items() if k in op._known_arguments}
    # Sanity check
    if grid.dim == 1 or len(blockshape) == 0:
        assert len(blocksizes) == 0
        return {}
    try:
        if dle[1].get('blockinner'):
            assert len(blocksizes) >= 1
            if grid.dim == len(blockshape):
                assert len(blocksizes) == len(blockshape)
            else:
                assert len(blocksizes) <= len(blockshape)
        return blocksizes
    except AttributeError:
        assert len(blocksizes) == 0
        return {}

shape = (50, 50, 50)
spacing = (10., 10., 10.)

parser = ArgumentParser(description="Some parser")
parser.add_argument('-b', dest='blocksize', default=4, type=int,
                        help="Block size")

args = parser.parse_args()


tn=500

blockinner = True

blocking_params = {'dle':('blocking', {'openmp': True, 'blockalways': True,
                                      'blockinner': blockinner})}

solver = acoustic_setup(shape, spacing, tn, **blocking_params)

sizes = list(range(1, 64))

f_times = []
g_times = []

for blocksize in sizes:

    # blocksize = args.blocksize
    blockshape = (blocksize, blocksize, blocksize)
    blocksizes = get_blocksizes(solver.op_fwd(), blocking_params['dle'],
                                solver.model.grid, blockshape)

    rec, u, fsummary = solver.forward(save=True, **blocksizes)

    fw_time = sum([fsummary[s].time for s in fsummary])

    grad, gsummary = solver.gradient(rec, u, **blocksizes)

    g_time = sum([gsummary[s].time for s in gsummary])

    print("%d, %f, %f" % (blocksize, fw_time, g_time))
    f_times.append(fw_time)
    g_times.append(g_time)

plt.plot(sizes, f_times, label="Forward")
plt.plot(sizes, g_times, label="Adjoint")
plt.xlabel("Block size")
plt.ylabel("Run time(s)")
plt.title("Run times for varying block sizes")
plt.legend()

plt.savefig("timings.pdf")



