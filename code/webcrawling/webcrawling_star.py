
import os
import json

def star_range_distributed():
    source_dir = ["asiayo_comments", "easycamp_comments", "klook_comments"]

    total_star_1_cnt = 0
    total_star_2_cnt = 0
    total_star_3_cnt = 0
    total_star_4_cnt = 0
    total_star_5_cnt = 0


    star_1_cnt = 0
    star_2_cnt = 0
    star_3_cnt = 0
    star_4_cnt = 0
    star_5_cnt = 0

    for source in source_dir:
        dir_path = os.path.join(os.getcwd(), "data\\{}".format(source))
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                print(file_path)
                f = open(file_path, encoding="utf-8-sig")
                data = json.load(f)
                f.close()
                print(len(data["comments"]))
                data["comments"] = list(filter(lambda x: len(x["content"]) > 10, data["comments"]))
                print(len(data["comments"]))
                for c in data["comments"]:
                    if c["rating"] == 1:
                        star_1_cnt += 1
                        total_star_1_cnt += 1
                    elif c["rating"] == 2:
                        star_2_cnt += 1
                        total_star_2_cnt += 1
                    elif c["rating"] == 3:
                        star_3_cnt += 1
                        total_star_3_cnt += 1
                    elif c["rating"] == 4:
                        star_4_cnt += 1
                        total_star_4_cnt += 1
                    elif c["rating"] == 5:
                        star_5_cnt += 1
                        total_star_5_cnt += 1

                with open(file_path, 'w', encoding="utf-8-sig") as f:
                    json.dump(data, f, indent=4, ensure_ascii=False)

        # save_path = os.path.join(os.getcwd(), "data\\star_distributed")
        # file_name = os.path.join(save_path, 'type_{}_star_.json'.format(t))
        # with open(file_name, 'w', encoding="utf-8-sig") as f:
        #     star_distributed = {
        #         "star_1": star_1_cnt,
        #         "star_2": star_2_cnt,
        #         "star_3": star_3_cnt,
        #         "star_4":star_4_cnt,
        #         "star_5":star_5_cnt,
        #     }
        #     json.dump(star_distributed, f, indent=4, ensure_ascii=False)

    save_path = os.path.join(os.getcwd(), "data\\star_distributed")
    file_name = os.path.join(save_path, 'all_star.json')
    with open(file_name, 'w', encoding="utf-8-sig") as f:
        star_distributed = {
            "star_1": total_star_1_cnt,
            "star_2": total_star_2_cnt,
            "star_3": total_star_3_cnt,
            "star_4": total_star_4_cnt,
            "star_5": total_star_5_cnt,
        }
        json.dump(star_distributed, f, indent=4, ensure_ascii=False)


def google_star_range_distributed():
    star_1_cnt = 0
    star_2_cnt = 0
    star_3_cnt = 0
    star_4_cnt = 0
    star_5_cnt = 0

    dir_path = os.path.join(os.getcwd(), "data\\google_comments")
    for file in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file)
        f = open(file_path, encoding="utf-8-sig")
        data = json.load(f)
        f.close()

        data["comments"] = list(filter(lambda x: len(x["content"]) > 10, data["comments"]))
        for c in data["comments"]:
            if c["rating"] == 1:
                star_1_cnt += 1
            elif c["rating"] == 2:
                star_2_cnt += 1
            elif c["rating"] == 3:
                star_3_cnt += 1
            elif c["rating"] == 4:
                star_4_cnt += 1
            elif c["rating"] == 5:
                star_5_cnt += 1

        with open(file_path, 'w', encoding="utf-8-sig") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

        save_path = os.path.join(os.getcwd(), "data\\star_distributed")
        file_name = os.path.join(save_path, 'type_google_star_.json')
        with open(file_name, 'w', encoding="utf-8-sig") as f:
            star_distributed = {
                "star_1": star_1_cnt,
                "star_2": star_2_cnt,
                "star_3": star_3_cnt,
                "star_4":star_4_cnt,
                "star_5":star_5_cnt,
            }
            json.dump(star_distributed, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    star_range_distributed()
    google_star_range_distributed()