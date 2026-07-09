configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(SYM_TEST_DEPENDS
  sym-opt
  reloc-plan-builder-test
  libreloc-test
)

add_lit_testsuite(check-sym "Running Sym regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${SYM_TEST_DEPENDS}
)

set_target_properties(check-sym PROPERTIES FOLDER "Tests")
