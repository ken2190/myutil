def salaryScale(sal):
    if sal >= 4000:
        return 'high'
    elif sal > 2500 and sal < 4000:
        return 'medium'
    else:
        return 'low'

salaryScaleUDF = F.udf(salaryScale, T.StringType())

df_emp.withColumn("SalaryScale", salaryScaleUDF("sal")).show()
+-----+------+---------+----+----------+----+----+------+-----------+
|empno| ename|      job| mgr|  hiredate| sal|comm|deptno|SalaryScale|
+-----+------+---------+----+----------+----+----+------+-----------+
| 7369| SMITH|    CLERK|7902|17-12-1980| 800|null|    20|        low|
| 7499| ALLEN| SALESMAN|7698| 20-2-1981|1600| 300|    30|        low|
| 7521|  WARD| SALESMAN|7698| 22-2-1981|1250| 500|    30|        low|
| 7566| JONES|  MANAGER|7839|  2-4-1981|2975|null|    20|     medium|
| 7654|MARTIN| SALESMAN|7698| 28-9-1981|1250|1400|    30|        low|
| 7698| BLAKE|  MANAGER|7839|  1-5-1981|2850|null|    30|     medium|
| 7782| CLARK|  MANAGER|7839|  9-6-1981|2450|null|    10|        low|
| 7788| SCOTT|  ANALYST|7566| 13-JUL-87|3000|null|    20|     medium|
| 7839|  KING|PRESIDENT|null|17-11-1981|5000|null|    10|       high|
| 7844|TURNER| SALESMAN|7698|  8-9-1981|1500|   0|    30|        low|
| 7876| ADAMS|    CLERK|7788| 13-JUL-87|1100|null|    20|        low|
| 7900| JAMES|    CLERK|7698| 3-12-1981| 950|null|    30|        low|
| 7902|  FORD|  ANALYST|7566| 3-12-1981|3000|null|    20|     medium|
| 7934|MILLER|    CLERK|7782| 23-1-1982|1300|null|    10|        low|
| 9999|  ANDY|      DBA|null| 02-1-1981|4300|null|  null|       high|
+-----+------+---------+----+----------+----+----+------+-----------+