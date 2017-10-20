# Bash script to create a randomized selection of a given dataset
# Split the given dataset in path_to_source + source_folder into a randomized dataset in path_to_target

path_to_source=$1
path_to_target=$2
split_size=$3

rm -r $path_to_target;
mkdir $path_to_target;

for dir in $(ls $path_to_source)
do
    if [ -d "$path_to_source/$dir" ]
    then
        mkdir -p $path_to_target/validation/$dir;
        mkdir -p $path_to_target/train/$dir;
        for sdir in $(ls $path_to_source/$dir)
        do        
            k=$(($RANDOM % 100));
            if [ $k -ge $split_size ]
            then    
                cp -r $path_to_source/$dir/$sdir $path_to_target/validation/$dir/$sdir;
            else
                cp -r $path_to_source/$dir/$sdir $path_to_target/train/$dir/$sdir;
            fi
        done
    fi
done

