#find . -size +100M | cat >> .gitignore
#find . -size +100M | sed 's|^\./||g' | cat >> .gitignore; awk '!NF || !seen[$0]++' .gitignore
git add .gitignore
git rm -r --cached .
git add .
