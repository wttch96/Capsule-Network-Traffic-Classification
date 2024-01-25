from util.data.USTC import USTCPreprocess

if __name__ == "__main__":
    preprocess = USTCPreprocess("USTC-TFC2016", "USTC-TFC2016-IMAGE", 20, 1100, 200, 110)
    preprocess.preprocess()
