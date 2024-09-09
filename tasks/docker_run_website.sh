#!/bin/bash
# Run from the root of the project
docker run --rm -v $(pwd):/srv/jekyll -p 4000:4000 -e PAGES_REPO_NWO=serval-uni-lu/tabularbench jekyll/jekyll:pages sh -c "gem install webrick && jekyll serve --source ./website --destination ./website/_site"
