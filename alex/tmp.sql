SELECT 
    *,
    -- 将原始时间字符串转换为MySQL datetime类型，用于验证转换结果
    STR_TO_DATE(
        SUBSTRING_INDEX(create_time, ' GMT', 1),  -- 截取到GMT前的部分
        '%a %b %d %Y %H:%i:%s'  -- 匹配 "Wed Sep 11 2024 10:35:08" 格式
    ) AS converted_create_time
FROM task2
-- 按转换后的时间降序排序
ORDER BY STR_TO_DATE(
    SUBSTRING_INDEX(create_time, ' GMT', 1),
    '%a %b %d %Y %H:%i:%s'
) DESC limit 10;


UPDATE task2
SET 
    task_name = '图纸plan_2的识别',
    task_type = 'All'
WHERE task_id = '2c34a2b5-88c3-4d78-a42b-5623cf225044';
