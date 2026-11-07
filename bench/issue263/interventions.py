"""Controlled diagnostic bypasses; these are not production optimizations."""


class Intervention:
    def __init__(self, name='baseline', *, qualified_metadata=False):
        if name not in ('baseline', 'no_report', 'no_check', 'both'):
            raise ValueError(f'unknown intervention: {name}')
        if name in ('no_check', 'both') and not qualified_metadata:
            raise ValueError('signature bypass requires the qualified metadata contract')
        self.name, self.undo = name, []

    def __enter__(self):
        from reloc_torch import fast
        if self.name in ('no_report', 'both'):
            original = fast.FastCompletion._finish
            def finish(completion):
                # Called only after native query/wait establishes completion.
                # Keep retirement and references identical; omit only report
                # dictionary construction and Python accounting.
                completion._done = True
                completion._native = None
            self.undo.append((fast.FastCompletion, '_finish', original))
            fast.FastCompletion._finish = finish
        if self.name in ('no_check', 'both'):
            original = fast._Program.check
            self.undo.append((fast._Program, 'check', original))
            fast._Program.check = lambda program, source: None
        return self

    def __exit__(self, *exc):
        for owner, name, original in reversed(self.undo):
            setattr(owner, name, original)
        self.undo.clear()

    def description(self):
        return dict(name=self.name,
                    report_accounting_enabled=self.name not in ('no_report', 'both'),
                    python_signature_checks_enabled=self.name not in ('no_check', 'both'),
                    native_validation_enabled=True, experimental_only=True)
