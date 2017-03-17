# Pull and push latest experiment results
# Alfredo Canziani, Mar 17

# Run it as
# ./update_experiments.sh

local=$(hostname)
case $local in
    "GPU0") remote=$GPU8;;
    "GPU8") remote=$GPU0;;
    *)      echo "Something's wrong"; exit -1;;
esac

echo; printf "%.s#" {1..80}; echo
echo -n "Getting experiments from $remote to $local"
echo; printf "%.s#" {1..80}; echo; echo
rsync \
    --update \
    --archive \
    --verbose \
    --human-readable \
    $remote:MatchNet/results/ \
    ../results

echo; printf "%.s#" {1..80}; echo
echo -n "Sending experiments to $remote from $local"
echo; printf "%.s#" {1..80}; echo; echo
rsync \
    --update \
    --archive \
    --verbose \
    --human-readable \
    ../results/ \
    $remote:MatchNet/results
