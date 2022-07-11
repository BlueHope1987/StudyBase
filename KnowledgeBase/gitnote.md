一些git操作

先init初始库
配置remote add origin usrnm:pswd@gitaddress.git
pull拉取分支 pull origin master
fetch拉取最新 push推送 merge合并分支
pull=fetch+merge clone克隆整个版本库

提交更改需要定义个人信息
git config --global user.name "John Doe"
git config --global user.email johndoe@example.com

github删除已经push到服务器上的commit的方法
使用两条指令：
git reset --hard <commit_id>
git push origin HEAD --force
其中commit_id是你想回到的commit的id（即想删除的commit的前一个commit），可以在github.com的commit中查看。

git给文件及文件夹改名
直接在vscode资源管理器改名会被源代码判定删除与新建，用git命令改名或可避免多余的改动。如将ReadyFor2021SA改名为ReadyForSA
git mv ReadyFor2021SA ReadyForSA
