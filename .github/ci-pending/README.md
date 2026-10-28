Move ci.yml back to .github/workflows/ once the GitHub token has the `workflow` scope
(`gh auth refresh -h github.com -s workflow`). GitHub rejects workflow files pushed without it.
