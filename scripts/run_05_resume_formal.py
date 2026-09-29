"""Zero-argument resume entrypoint for the declared Scheme-2 budget."""

from _bootstrap import formal_iterations, require_cuda, run

require_cuda()
run("scripts/scheme2.py", "--resume", "--iterations", str(formal_iterations()))
