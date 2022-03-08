#!/bin/bash
set -x
################################################################################
# File:    buildDocs.sh
# Purpose: Script that builds our documentation using sphinx and updates GitHub
#          Pages. This script is executed by:
#            .github/workflows/docs_pages_workflow.yml
#
# Authors: Michael Altfield <michael@michaelaltfield.net>
# Created: 2020-07-17
# Updated: 2020-07-20
# Version: 0.2
################################################################################
 
###################
# INSTALL DEPENDS #
###################
 
apt-get update
apt-get -y install git rsync python3-sphinx python3-sphinx-rtd-theme python3-stemmer python3-git python3-pip python3-virtualenv python3-setuptools
 
python3 -m pip install --upgrade rinohtype pygments
 
#####################
# DECLARE VARIABLES #
#####################
 
pwd
ls -lah
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
 
# make a new temp dir which will be our GitHub Pages docroot
docroot=`mktemp -d`

export REPO_NAME="${GITHUB_REPOSITORY##*/}"
 
##############
# BUILD DOCS #
##############
 
# first, cleanup any old builds' static assets
make -C docs clean
 
# get a list of branches, excluding 'HEAD' and 'gh-pages'
versions="`git for-each-ref '--format=%(refname:lstrip=-1)' refs/remotes/origin/ | grep -viE '^(HEAD|gh-pages)$'`"
for current_version in ${versions}; do
 
   # make the current language available to conf.py
   export current_version
   git checkout ${current_version}
 
   echo "INFO: Building sites for ${current_version}"
 
   # skip this branch if it doesn't have our docs dir & sphinx config
   if [ ! -e 'docs_ghpage/conf.py' ]; then
      echo -e "\tINFO: Couldn't find 'docs_ghpage/conf.py' (skipped)"
      continue
   fi

   #### Install utilmy
   pip install -e .


 
   # languages="en `find docs_ghpage/locales/ -mindepth 1 -maxdepth 1 -type d -exec basename '{}' \;`"
   languages="en"

   for current_language in ${languages}; do
 
      # make the current language available to conf.py
      export current_language
 
      ##########
      # BUILDS #
      ##########
      echo "INFO: Building for ${current_language}"
 

# Running Sphinx v4.4.0
# Configuration error:
# config directory doesn't contain a conf.py file (/__w/myutil/myutil/utilmy)
# + rsync -av docs_ghpage/_build/html/ /tmp/tmp.EHpPcy6lE3/
# rsync: change_dir "/__w/myutil/myutil//docs_ghpage/_build/html" failed: No such file or directory (2)
# sending incremental file list
# rsync error: some files/attrs were not transferred (see previous errors) (code 23) at main.c(1207) [sender=3.1.3]

      # Generate the RST
      sphinx-apidoc -f -o docs_ghpage/_build/rst/${current_language}/${current_version}/   utilmy/


      # Generate HTML  from RST
      sphinx-build -b html docs_ghpage/_build/rst/${current_language}/${current_version}/  docs_ghpage/_build/html/${current_language}/${current_version} -D language="${current_language}"
 

      # PDF #
      # sphinx-build -b rinoh utilmy/ docs_ghpage/_build/rinoh -D language="${current_language}"
      # mkdir -p "${docroot}/${current_language}/${current_version}"
      # cp "docs_ghpage/_build/rinoh/target.pdf" "${docroot}/${current_language}/${current_version}/helloWorld-docs_${current_language}_${current_version}.pdf"
 
      # EPUB #
      # sphinx-build -b epub utilmy/ docs_ghpage/_build/epub -D language="${current_language}"
      # mkdir -p "${docroot}/${current_language}/${current_version}"
      # cp "docs_ghpage/_build/epub/target.epub" "${docroot}/${current_language}/${current_version}/helloWorld-docs_${current_language}_${current_version}.epub"
 
      # copy the static assets produced by the above build into our docroot
      rsync -av "docs_ghpage/_build/html/" "${docroot}/"
 
   done
 
done
 
# return to master branch
git checkout master
 
#######################
# Update GitHub Pages #
#######################
 
git config --global user.name "${GITHUB_ACTOR}"
git config --global user.email "${GITHUB_ACTOR}@users.noreply.github.com"
 
pushd "${docroot}"
 
# don't bother maintaining history; just generate fresh
git init
git remote add deploy "https://token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
git checkout -b gh-pages
 
# add .nojekyll to the root so that github won't 404 on content added to dirs
# that start with an underscore (_), such as our "_content" dir..
touch .nojekyll
 
# add redirect from the docroot to our default docs language/version
cat > index.html <<EOF
<!DOCTYPE html>
<html>
   <head>
      <title>helloWorld Docs</title>
      <meta http-equiv = "refresh" content="0; url='/${REPO_NAME}/en/main/'" />
   </head>
   <body>
      <p>Please wait while you're redirected to our <a href="/${REPO_NAME}/en/main/">documentation</a>.</p>
   </body>
</html>
EOF
 
# Add README
cat > README.md <<EOF
# GitHub Pages Cache
 
Nothing to see here. The contents of this branch are essentially a cache that's not intended to be viewed on github.com.
 
 
If you're looking to update our documentation, check the relevant development branch's 'docs_ghpage/' dir.
 
For more information on how this documentation is built using Sphinx, Read the Docs, and GitHub Actions/Pages, see:
 
 * https://tech.michaelaltfield.net/2020/07/18/sphinx-rtd-github-pages-1
EOF
 
# copy the resulting html pages built from sphinx above to our new git repo
git add .
 
# commit all the new files
msg="Updating Docs for commit ${GITHUB_SHA} made on `date -d"@${SOURCE_DATE_EPOCH}" --iso-8601=seconds` from ${GITHUB_REF} by ${GITHUB_ACTOR}"
git commit -am "${msg}"
 
# overwrite the contents of the gh-pages branch on our github.com repo
git push deploy gh-pages --force
 
popd # return to main repo sandbox root
 
# exit cleanly
exit 0
