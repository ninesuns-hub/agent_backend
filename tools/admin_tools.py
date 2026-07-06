from .base import Tool

def create_admin_tool(query_func):
    """
    创建课程事务查询工具
    """
    return Tool(
        name="query_course_admin",
        func=query_func,
        description="非常有用！当你需要查询关于离散数学课程的行政事务时请调用。你应该根据用户问题提取出一个最相关的类别或关键词作为输入。可选类别包括：[teacher(授课教师信息/联系方式), grading(评分标准/考试规定), schedule(教学大纲/章节课时), info(课程编号/学分/地点/答疑时间), rule(课堂规定/网站地址)]。"
    )
