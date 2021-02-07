一些git操作

先init初始库
配置remote add origin usrnm:pswd@gitaddress.git
pull拉取分支 pull origin master
fetch拉取最新 push推送 merge合并分支
pull=fetch+merge clone克隆整个版本库

github删除已经push到服务器上的commit的方法
使用两条指令：
git reset --hard <commit_id>
git push origin HEAD --force
其中commit_id是你想回到的commit的id（即想删除的commit的前一个commit），可以在github.com的commit中查看。