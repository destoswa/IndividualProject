import concurrent.futures
import time
from tqdm import tqdm


def do_wait(secs):
    time.sleep(secs)
    return f"slept for {secs} secs"


def main():
    list_secs = range(10)
    time_sart = time.time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(do_wait, list_secs), total=len(list_secs)))

        for result in results:
            print(result)

    time_end = time.time() - time_sart
    print(f"Total duration was : {round(time_end, 2)}")


if __name__ == '__main__':
    main()
