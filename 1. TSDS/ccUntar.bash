#! /bin/bash

cd ./2015
for FILE in \`find *.tar.gz\`
do
IFS='.' read -r -a array <<< \"\$FILE\"
IFS='-' read -r -a cc <<< \"\${array[0]}\"
mkdir \"\${cc[0]}\"
tar xzf \$FILE -C ./\"\${cc[0]}\"
done
exit 0