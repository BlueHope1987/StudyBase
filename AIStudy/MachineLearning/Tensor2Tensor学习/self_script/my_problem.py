# coding=utf-8
from tensor2tensor.utils import registry
from tensor2tensor.data_generators import problem, text_problems

#自定义的problem一定要加该装饰器，不然t2t库找不到自定义的problem
@registry.register_problem
class MyProblem(text_problems.Text2TextProblem):
    @property
    def approx_vocab_size(self):
        return 2**11

    @property
    def is_generate_per_split(self):
        return False

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split
		#读取原始的训练样本数据
        q_r = open("./rawdata/q.txt", "r")
        a_r = open("./rawdata/a.txt", "r")

        comment_list = q_r.readlines()
        tag_list = a_r.readlines()
        q_r.close()
        a_r.close()
        for comment, tag in zip(comment_list, tag_list):
            comment = comment.strip()
            tag = tag.strip()
            yield {
                "inputs": comment,
                "targets": tag
            }
#————————————————
#版权声明：本文为CSDN博主「csa121」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
#原文链接：https://blog.csdn.net/csa121/article/details/79605215