from _bootstrap import formal_iterations, require_cuda, run

require_cuda()
run("scripts/scheme2.py", "--run", "--iterations", str(formal_iterations()))
