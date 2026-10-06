**Variable-level audit of the primary dataset (before cleaning).**

| Variable           | Type        |   Missing (n) |   Missing (%) |   Unique |   Min |   Max | Levels                                     |
|:-------------------|:------------|--------------:|--------------:|---------:|------:|------:|:-------------------------------------------|
| Age                | numeric     |             0 |          0    |        7 |  12   |  18   |                                            |
| Grade              | numeric     |             0 |          0    |        5 |   8   |  12   |                                            |
| Gender             | categorical |             0 |          0    |        3 |       |       | Male; Female; Prefer not to say            |
| ParentEducation    | categorical |             0 |          0    |        4 |       |       | High school; College; Postgrad; No formal  |
| IncomeRange        | categorical |             0 |          0    |        3 |       |       | Low; Medium; High                          |
| StudySpace         | categorical |             0 |          0    |        2 |       |       | No; Yes                                    |
| StudyHoursPerWeek  | numeric     |             0 |          0    |      182 |   1   |  23.3 |                                            |
| SleepHoursPerNight | numeric     |             0 |          0    |       60 |   3.9 |  11.6 |                                            |
| ScreenTimeDaily    | numeric     |             0 |          0    |       75 |   1   |   8.9 |                                            |
| Motivation(1-5)    | numeric     |             0 |          0    |        5 |   1   |   5   |                                            |
| Stress(1-5)        | numeric     |             0 |          0    |        5 |   1   |   5   |                                            |
| Extracurricular    | categorical |             0 |          0    |        2 |       |       | No; Yes                                    |
| HighAttendance     | categorical |             0 |          0    |        2 |       |       | Yes; No                                    |
| LastTermPercentage | numeric     |             0 |          0    |      448 |  34.3 | 100   |                                            |
| DifficultSubject   | categorical |           124 |         10.26 |        5 |       |       | History; Math; Science; Geography; English |

*Note.* DifficultSubject has missing responses; these are coded as an explicit 'Unknown' level.