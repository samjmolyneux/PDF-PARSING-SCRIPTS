# Documentation guidance

Follow the style of [Flowde's documentation](https://github.com/EPPI-Centre/Flowde/blob/main/docs/AGENTS.md).

- Keep three main pages: Start here, Run parsers, and Deploy parsers.
- Lead guides with a short introduction and working examples. Explain options
  and less common workflows afterwards.
- Keep the running quick start sufficient for ordinary users. Deployment and
  image-build details belong in the administrator guide.
- Use fenced code blocks without shell prompts so readers can copy commands.
  Separate alternative commands into separate blocks and identify placeholders.
- State defaults, requirements and example choices explicitly. Use active voice
  and name the relevant command, file or Azure resource.
- Check commands and defaults against the current code. Distinguish local
  preview commands from commands with `--apply`, and local validation from a
  successful Azure image build or GPU run.
- Retain the Read the Docs theme and code-copy buttons. Run
  `python -m mkdocs build --strict` after documentation changes.
