# Check what has changed across experiments
# Alfredo Canziani, Mar 17

# ./check_exp_diff.sh base_exp new_exp

base_exp="../results/$1/train.log"
new_exp="../results/$2/train.log"

git diff --no-index --word-diff --color=always $base_exp $new_exp | head -7

echo ""
hash=$(awk -F ": " '/hash/{print $2}' $new_exp)
git show --stat $hash
