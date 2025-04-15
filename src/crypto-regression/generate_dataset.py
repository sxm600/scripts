import os
import sys
import argparse
import pandas as pd


TARGET_COL = 'Low'

parser = argparse.ArgumentParser(description="Generate dataset for cryptocurrencies.")

parser.add_argument('--out_path', '-o', type=str, default='dataset.csv',
                    help="Specify an output file path.")

parser.add_argument('in_paths', nargs='*',
                    help="Specify input file paths.")


def generate_dataframe(in_paths: list[str]) -> pd.DataFrame:
    df_out = pd.DataFrame()

    for in_path in in_paths:
        try:
            df_in = pd.read_csv(in_path)
        except Exception:
            print(f'Failed to read {in_path} as csv.')
            sys.exit()

        if TARGET_COL not in df_in.columns:
            print(f'Column "{TARGET_COL}" was not found in {in_path} dataset.')
            sys.exit()

        *folders, file_name = in_path.split('/')
        df_out[file_name.removesuffix('.csv')] = df_in[TARGET_COL]

    return df_out


def main():
    args = parser.parse_args()

    if not args.in_paths:
        print('No input dataframes provided. Write "-h" or --help for manual.')
        sys.exit()

    for in_path in args.in_paths:
        if not os.path.isfile(in_path):
            print(f"There is no such file destination as '{in_path}'")
            sys.exit()

    df = generate_dataframe(args.in_paths)

    print(f'Writing dataframe to {args.out_path}')
    df.to_csv(args.out_path, index=False)

    print(f'Successfully saved dataframe to {args.out_path}')


if __name__ == '__main__':
    main()
