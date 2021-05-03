data_folder="data" 
output_path="output/20210428"

n_mid="10 100 300 500" #100
eta="0.1 0.01 0.001" # 0.01
dropout_rate="0.8 0.5 0.2 0.0" # 0.5

batch_size="1 100 500 1000" # 100
normalisation="Min-Max z-score log" # Min-Max

# 其他參數固定
for N in $normalisation;
do
  output_folder="$output_path"/Nm100_E0.01_B100_Dr0.5_Nor"$N"
  python train.py --normalisation $N --output_folder $output_folder
done

for B in $batch_size;
do
  output_folder="$output_path"/Nm100_E0.01_B"$B"_Dr0.5_NorMin-Max
  python train.py --batch_size $B --output_folder $output_folder
done

# 全部參數組合共 48 組
for E in $eta;
do
  for M in $n_mid;
  do
    for D in $dropout_rate;
    do
      output_folder="$output_path"/Nm"$M"_E"$E"_B100_Dr"$D"_NorMin-Max
      python train.py --data_folder "$data_folder" \
                        --output_folder "$output_folder" \
                        --n_mid "$M" \
                        --eta "$E" \
                        --batch_size "$B" \
                        --dropout_rate "$D"
    done
  done
done