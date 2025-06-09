# 定义窗口配置数组
$windows = @(
    @{
        Title = "redis"
        Path = "E:\School\Grad1\CAD\MyCAD2\redis"
        Commands = @(
            "redis-server.exe redis.windows.conf"
        )
    },
    @{
        Title = "celery"
        Path = "E:\School\Grad1\CAD\MyCAD2\CAD-main\services2"
        Commands = @(
            "conda activate car_py38",
            "celery -A main_server4.celery worker --loglevel=info -P eventlet"
        )
    },
    @{
        Title = "server4"
        Path = "E:\School\Grad1\CAD\MyCAD2\CAD-main\services2"
        Commands = @(
            "conda activate car_py38",
            "python main_server4.py"
        )
    },
    @{
        Title = "server_cad"
        Path = "E:\School\Grad1\CAD\MyCAD2\CAD-main\services2"
        Commands = @(
            "conda activate cad_py38",
            "python main_server_cad1.py"
        )
    },
    @{
        Title = "npm"
        Path = "E:\School\Grad1\CAD\MyCAD2\vue-element-admin-v3\vue-element-admin"
        Commands = @(
            "npm run dev"
        )
    }
)

# 遍历创建每个 CMD 窗口
foreach ($win in $windows) {
    # 组合命令（用 && 连接，确保按顺序执行）
    $commandString = $win.Commands -join " && "
    
    # 启动 CMD 进程并设置标题
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k title $($win.Title) && cd /d $($win.Path) && $commandString"
}