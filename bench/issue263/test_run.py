from bench.issue263.run import clean_environment


def test_profiler_launch_does_not_inherit_unrelated_credentials_or_sessions():
    inherited={'HOME':'/home/example','PATH':'/usr/bin','LANG':'C.UTF-8','GH_TOKEN':'private-value',
               'SSH_CONNECTION':'private-addresses','VSCODE_GIT_IPC_HANDLE':'private-socket',
               'COPILOT_DEBUG_NONCE':'private-session'}
    assert clean_environment(inherited)=={'HOME':'/home/example','PATH':'/usr/bin','LANG':'C.UTF-8'}
