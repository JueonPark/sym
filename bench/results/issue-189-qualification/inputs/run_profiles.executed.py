import os, subprocess
from pathlib import Path
root=Path('/tmp/sym-pinning-results/traces')
env=dict(os.environ, PYTHONPATH='/tmp/sym-pinning-trace-build/python:/tmp/sym-pinning-qualification/libreloc/python', SYM_RELOC_EXPORT='/tmp/sym-pinning-build/sym/tools/sym-reloc-export', SYM_OPT='/tmp/sym-pinning-build/sym/tools/sym-opt', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
python='/tmp/sym-torch-cuda/bin/python'; nsys='/tmp/sym-cuda-toolkit-12.6.3/bin/nsys'
script='/tmp/sym-pinning-qualification/bench/issue189/profile_pinning.py'
for policy,buffers in [('configured',4),('configured',1),('unconfigured',4)]:
 name=f'{policy}-{buffers}'; prefix=str(root/name)
 commands=[
  [nsys,'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--force-overwrite=true','--output',prefix,'taskset','-c','4-7,20-23',python,script,'--policy',policy,'--buffers',str(buffers),'--output',prefix+'.run.json'],
  [nsys,'export','--type','sqlite','--force-overwrite=true','--output',prefix+'.sqlite',prefix+'.nsys-rep'],
  [python,script,'--policy',policy,'--buffers',str(buffers),'--sqlite',prefix+'.sqlite','--output',prefix+'.analysis.json']]
 with (root/(name+'.log')).open('w') as log:
  for command in commands:
   log.write(repr(command)+'\n');log.flush()
   subprocess.run(command,env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
 print(name,'passed',flush=True)
