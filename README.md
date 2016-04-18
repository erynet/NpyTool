# NpyTool

usage: npy_export [-h] (-p | -d) [-P PATH] [-f FNMATCH_PATTERN] [-R]
                  [-s LENGTH_OF_SIDE] [--mysqlhost MYSQLHOST]
                  [--mysqlport MYSQLPORT] [--mysqluser MYSQLUSER]
                  [--mysqlpasswd MYSQLPASSWD] [-i INDEX]
                  [-w WORKER_PROCESS_COUNT] [-c] [-o OUTFILE]
                  [-e MAX_ENTRY_COUNT] [-a AUGMENTATION] [-pp PREPROCESSING]
                  [-l LOGTO] [-L LOGLEVEL]

optional arguments:
  -h, --help            show this help message and exit
  -p, --frompath
  -d, --fromdb
  -w WORKER_PROCESS_COUNT, --worker_process_count WORKER_PROCESS_COUNT
  -c, --compress
  -o OUTFILE, --outfile OUTFILE
  -e MAX_ENTRY_COUNT, --max_entry_count MAX_ENTRY_COUNT
  -a AUGMENTATION, --augmentation AUGMENTATION
  -pp PREPROCESSING, --preprocessing PREPROCESSING
  -l LOGTO, --logto LOGTO
  -L LOGLEVEL, --loglevel LOGLEVEL

From File:
  -P PATH, --path PATH
  -f FNMATCH_PATTERN, --fnmatch_pattern FNMATCH_PATTERN
  -R, --recursive
  -s LENGTH_OF_SIDE, --length_of_side LENGTH_OF_SIDE

From Database:
  --mysqlhost MYSQLHOST
  --mysqlport MYSQLPORT
  --mysqluser MYSQLUSER
  --mysqlpasswd MYSQLPASSWD
  -i INDEX, --index INDEX



# Example

python npy_export.py 
  --frompath 
  --path /ml/dataset/7_OPEJ_Gray/ 
  --recursive 
  --fnmatch_pattern "*.jpg" 
  --outfile test8 
  --loglevel INFO 
  --augmentation "rtf" 
  --preprocessing "histeq,pcaw,gaussian_noise  0.5" 
  --worker_process_count 4 
  --max_entry_count 500000000000
  

