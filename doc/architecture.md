# Data

Elements:
- activity
- summary
- recordings

Relations:
- activity <-> summary
- activity --> recordings

# Components

Elements:
- storage
- CLI
- web_backend
- web_frontend

Relations:
- storage <-> CLI
- storage --> web_backend <-> web_frontend
