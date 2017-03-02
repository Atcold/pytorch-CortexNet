# Prepares environment for new experiment
# Alfredo Canziani, Mar 17

# It expects a directory / link named "results" in the cwd containing
# numerically increasing folders with 3 digits

old_CLI=$(awk -F': ' '/CLI/{print $2}' last/train.log)
max_exp=$(ls results/ | tail -1)
dst_path=$(printf "results/%03d" "$((++max_exp))")

echo -n " > Creating folder: "
mkdir $dst_path
ls -d --color=always $dst_path

echo -n " > Linking:"
ln -snf $dst_path last
ls -l --color=always last | awk -F"$USER" '{print $3}'

echo " > Previously you've used the following options"
echo "   python -u main.py $old_CLI | tee last/train.log"
