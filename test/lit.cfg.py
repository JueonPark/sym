import os
import lit.formats
import lit.util

config.name = "Sym"
config.test_format = lit.formats.ShTest(True)
config.suffixes = ['.mlir', '.test']
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.sym_build_dir, 'test')

# Add sym-opt to path
config.environment['PATH'] = os.path.pathsep.join([
    os.path.join(config.sym_build_dir, 'sym', 'tools'),
    config.environment.get('PATH', '')
])

# Add unit-test binaries to path
config.environment['PATH'] = os.path.pathsep.join([
    os.path.join(config.sym_build_dir, 'unittest', 'reloc'),
    os.path.join(config.sym_build_dir, 'libreloc', 'test'),
    config.environment.get('PATH', '')
])

# Add llvm tools (FileCheck, etc.)
config.environment['PATH'] = os.path.pathsep.join([
    config.llvm_tools_dir,
    config.environment.get('PATH', '')
])

config.substitutions.append(('%PATH%', config.environment['PATH']))
