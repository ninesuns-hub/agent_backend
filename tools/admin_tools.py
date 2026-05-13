from .base import Tool

def create_admin_tool(query_func):
    """
    创建课程事务查询工具
    """
    return Tool(
        name="query_course_admin",
        func=query_func,
        description="非常有用！当你需要查询关于离散数学课程的考试安排、作业要求、评分标准、老师联系方式、上课时间和地点等行政事务时，请调用此工具。输入应该是学生的原始问题。"
    )
