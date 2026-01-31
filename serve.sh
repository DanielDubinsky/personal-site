#!/bin/sh

# watch files and auto reload mkdocs server on changes
find docs -type f -name '*.md' -o -name 'mkdocs.yml' | entr -r sh -c 'pkill -f "^mkdocs serve" || true; ./venv/bin/mkdocs serve -a 127.0.0.1:8000'