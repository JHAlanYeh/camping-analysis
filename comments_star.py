
import os
import json

directory = "clean_data"

star_distributed = {
    "1-1": 0,
    "1-2": 0,
    "1-3": 0,
    "1-4": 0,
    "1-5": 0,
    "2-1": 0,
    "2-2": 0,
    "2-3": 0,
    "2-4": 0,
    "2-5": 0,
}

def star_range_distributed():
    if os.path.isdir(directory):
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            print(file_path)
            f = open(file_path, encoding="utf-8-sig")
            data = json.load(f)
            f.close()
           
            data["comments"] = list(filter(lambda x: len(x["content"]) > 10, data["comments"]))
            # print(len(data["comments"]))

            # data["comments"] = list(filter(lambda x: x["publishedDate"] > '2023/05/01', data["comments"]))

            # print(len(data["comments"]))
            for c in data["comments"]:
                if "T" in c["publishedDate"]:
                    c["publishedDate"] = c["publishedDate"].split("T")[0].replace("-", "/")

                if " " in c["publishedDate"]:
                    c["publishedDate"] = c["publishedDate"].split(" ")[0].replace("-", "/")


                if "type" not in c:
                    if data["type"] in [1, 2]:
                        c["type"] = data["type"]
                    else:
                        c["type"] = 0

                if c["type"] in [1, 2]:
                    star_distributed["{}-{}".format(str(c["type"]), str(c["rating"]))] += 1


            data["comments"] = list(filter(lambda x: x["publishedDate"] > '2018/01/01', data["comments"]))
            with open(file_path, 'w', encoding="utf-8-sig") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        print(star_distributed)
    # save_path = os.path.join(os.getcwd(), "data\\star_distributed")
    # file_name = os.path.join(save_path, 'all_star.json')
    # with open(file_name, 'w', encoding="utf-8-sig") as f:
    #     star_distributed = {
    #         "star_1": total_star_1_cnt,
    #         "star_2": total_star_2_cnt,
    #         "star_3": total_star_3_cnt,
    #         "star_4": total_star_4_cnt,
    #         "star_5": total_star_5_cnt,
    #     }
    #     json.dump(star_distributed, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    star_range_distributed()