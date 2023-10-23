find . -size +100M | cat >> .gitignore
git rm -r --cached .
git add .