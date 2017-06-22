# Bash script to create a randomized selection of a given dataset
# Split the given dataset in path_to_source + source_folder into a randomized dataset in path_to_target

nb_objects=$1
path_to_source=$2
path_to_target=$3

rm -r $path_to_target;
mkdir $path_to_target;

for dir in $(ls $path_to_source | shuf -n $nb_objects)
do
    if [ -d "$path_to_source/$dir" ]
    then
    	cp -r $path_to_source/$dir $path_to_target/$dir
    fi
done
