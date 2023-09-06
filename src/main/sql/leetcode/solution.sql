/* 595. 大的国家 */
SELECT name, population, area
FROM World
WHERE area >= 3000000 OR population >= 25000000;

/* 627. Swap Salary */
UPDATE Salary SET sex =
CASE sex
    WHEN 'm' THEN 'f'
    ELSE 'm'
END;

UPDATE Salary SET sex =
(ASCII(sex) ^ ASCII('m') ^ ASCII('f'));

/* 182. Duplicate Emails */
SELECT email
FROM person
GROUP BY email
HAVING COUNT(id) >= 2;

/* delete duplicate email */
# Write your MySQL query statement below
DELETE FROM person
WHERE id NOT IN (
    SELECT id
    FROM (
        SELECT min(id) AS id
        FROM person
        GROUP BY email
    ) AS m
);

DELETE p1
FROM Person p1, Person p2
WHERE p1.Email = p2.Email AND p1.Id > p2.Id;

/* 175. Combine Two Tables */
SELECT firstName, lastName, city, state
FROM Person LEFT JOIN Address ON personId;

/* 181. Employees Earning More Than Their Managers */
SELECT e1.name AS Employee
FROM Employee e1 INNER JOIN Employee e2 ON e1.managerId = e2.id
WHERE e1.salary >  e2.salary;

/* 183. Customers Who Never Order */
SELECT name as Customers
FROM Customers
WHERE id NOT IN (
    SELECT DISTINCT(customersId)
    FROM Orders
);

SELECT name AS Customers
FROM Customers LEFT JOIN Orders ON Customers.id = Orders.customersId
WHERE customersId IS NULL;

/* 184. 部门工资最高的员工 */
SELECT D.name AS Department, E.name AS Employee, E.salary AS Salary
FROM Employee E, Department D,
    (SELECT departmentId, MAX(salary) salary
     FROM Employee
     GROUP BY DepartmentId) M
WHERE E.departmentId = D.id AND E.departmentId = M.departmentId AND E.salary = M.salary;

/* 176. Second Highest Salary */
SELECT (
    SELECT DISTINCT Salary
    FROM Employee
    ORDER BY Salary DESC
    LIMIT 1, 1
) SecondHighestSalary;

/* 177. Nth Highest Salary */
CREATE FUNCTION getNthHighestSalary ( N INT ) RETURNS INT BEGIN

SET N = N - 1;
RETURN (
    SELECT (
        SELECT DISTINCT Salary
        FROM Employee
        ORDER BY Salary DESC
        LIMIT N, 1
    )
);

END

/* 178. Rank Scores */
SELECT S1.score AS score, COUNT(DISTINCT S2.score) AS rank
FROM Scores S1 INNER JOIN Scores S2 ON S1.score <= S2.score
GROUP BY S1.id, S1.score
ORDER BY S1.score DESC;

/* 180. Consecutive Numbers */
SELECT DISTINCT L1.num AS ConsecutiveNums
FROM Logs L1, Logs L2, Logs L3
WHERE L1.id = L2.id - 1
AND L2.id = L3.id - 1
AND L1.num = L2.num
AND L2.num = L3.num;









