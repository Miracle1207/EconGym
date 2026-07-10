"""
能力清单：Planner LLM 判断"EconGym 能不能建模某个问题"的唯一依据。

内容全部来自对 EconGym 源码的实际阅读（entities/*.py, agents/rule_based/*.py, cfg/base_config.yaml），
尤其是"哪些 YAML 参数在哪些算法下真正生效"这张表——这是文档完全没写、必须读代码才能发现的关键信息。
这份清单是人工维护的静态文本，不由 LLM 生成，目的是让 Planner 的每一次可行性判断都"有据可查"。
"""

CAPABILITY_MANIFEST = """\
# EconGym 能力清单

## 0. 关于"真实数据校准"的重要说明（判断可行性前必读）

EconGym 的默认初始状态**不是凭空生成的**，而是用真实美国数据校准的：
- households 的初始资产/收入/年龄/教育/劳动状态分布，是从 **2022 年美国 SCF（消费者金融调查）
  真实调查数据**（`agents/data/advanced_scfp2022_1110.csv`）按权重采样出来的。
- government 的初始 GDP、政府债务/GDP 比率、人口规模（`real_gdp`/`real_debt_rate`/`real_population`）
  是真实美国宏观数据。
- 因此，"以美国当前经济/人口状况为起点"的问题，**默认配置本身就是合理起点**，不需要用户额外提供数据，
  不应该因为"没有提供具体数据"就判定不可行。
- 人口动态方面：OLG 家庭模型的 `birth_rate`（出生率）可以配置；死亡率则是内置的、按年龄分层的固定
  经验表（`entities/households.py: calculate_death_probability`，不区分国家、不可配置）。
  内置任务 `population_aging.yaml`（Q3: How does aging impact the macroecony）就是专门做人口老龄化
  分析的，证明"模拟人口结构演化/老龄化趋势"是 EconGym 明确支持的场景。
- **真正的限制**是：EconGym 无法获取任何"真实世界的未来数据"（比如美国人口普查局对未来十年人口的
  官方预测值）——这不是 EconGym 的特有缺陷，任何仿真模型都无法"查到"尚未发生的真实数据。
  正确的做法是把这类问题**重新表述为一次模拟**：以真实美国数据为起点，在给定的出生率假设下，
  演化 N 年，观察模型自己模拟出的 `old_percent`/`dependency_ratio`/`house_age` 等指标趋势
  ——这是可行的，输出的是"本模型假设下的模拟趋势"，不是"美国官方人口预测"，解读时需要说明这个区别，
  但不能因此直接判定问题不可行。
- 同理，EconGym 也无法针对美国以外的其他具体国家做专门校准（没有其他国家的调查数据），
  这种情况才应该判定不可行或需要用户确认"是否接受用美国数据作为默认背景"。

## 1. households（家庭）

type 可选值：
- ramsey: 无限期寿命家庭，无年龄/退休概念，适合非代际问题
- OLG: 世代交叠模型，有年龄、退休、出生死亡，适合养老金/老龄化问题
- ramsey_risk_invest: 在 ramsey 基础上加入风险投资(股票)决策
- OLG_risk_invest: 在 OLG 基础上加入风险投资(股票)决策

可调参数（**是否生效取决于 house_alg，务必看清楚**）：
- households_n（家庭数量，1~5000）：始终生效
- CRRA（风险厌恶系数，0.1~5）：仅当 house_alg 属于 {ppo, ddpg, sac} 时才会真正驱动家庭行为；
  house_alg=rule_based/bc/data_based 时该参数只影响 reward 的计算方式，不影响任何实际决策
  （rule_based 用的是硬编码的按年龄经验分布采样，根本不读 CRRA）。
- IFE（跨期替代弹性倒数，0.1~5）：生效条件同 CRRA。
- stock_alpha（股价对买卖失衡的敏感度，0~1，仅 risk_invest 类型可用）：始终生效，与算法无关。
- birth_rate（仅 OLG，0~0.05）：始终生效，是**年化**出生率（如 0.011 = 1.1%/年），不是月度值。
- initial_working_age（仅 OLG，16~30）：始终生效。

**时间单位换算（容易配错，务必注意）**：EconGym 的每个 step 代表 **1 年**，不是 1 个月或其他单位
（OLG 家庭每个 step 年龄 `age += 1`；`years` 这个输出指标就是直接数 step 数）。
所以"模拟未来 N 年"应该设置 `episode_length = N`，**不要**按月折算成 `N * 12`。

## 2. government（政府；同一次仿真支持 1~3 个不同类型的政府同时存在）

**多政策协同**：governments 是一个列表，可以同时挂多个不同类型的政府来研究政策协同问题——
例如"财政+货币政策如何协同"用 [tax, central_bank]（对应内置任务 dbl_government），
"财政+货币+养老金三政协同"用 [tax, central_bank, pension]（对应内置任务 tre_government）。
注意：
- 同种类型的政府最多一个（不能挂两个 tax）。
- **列表顺序有意义：第一个政府是 main_gov**，家庭面对的所得税/养老金计算、GDP 等宏观指标
  都以它为准。把与问题最相关的政府放在第一位（例如问养老金问题时 pension 放第一）。
- 每个政府可以用 alg 字段单独指定自己的决策算法（如央行用 llm、财政用 rule_based），
  不填则统一用 algorithms.gov_alg。

- type=tax（财政当局）。gov_task 可选：gdp / gini / social_welfare / mean_welfare / gdp_gini。
  - tau/xi/tau_a/xi_a/Gt_prob（税率相关参数）：仅当 gov_alg 属于 {llm, ppo, ddpg} 时才会真正生效；
    gov_alg=rule_based 时每一步都会被硬编码覆盖为固定值(tau=0.263, xi=0.049, tau_a=0, xi_a=0, Gt_prob=0.189)，
    不管 YAML 里配了什么都没用。
  - consumption_tax_rate：环境级常量，**始终生效，与算法/家庭类型完全无关**——是"消费税类"
    问题最可靠的实验杠杆。
  - estate_tax_rate / estate_tax_exemption：环境级常量，**但只有当 households.type 包含 "OLG"
    时才会真正生效**！遗产税的计算（compute_estate_tax）只在家庭"死亡"事件发生时才会被触发，
    而 ramsey / ramsey_risk_invest 家庭是永生的、没有死亡逻辑，配了这两个参数也不会有任何效果。
    想问"遗产税/继承税"相关问题，households.type 必须选 OLG 或 OLG_risk_invest。
- type=pension（养老金当局）。**必须搭配 households.type 包含 "OLG"**，否则养老金机制完全不会被触发。
  - retire_age / contribution_rate：rule_based 下"部分生效"——会被当作 IMF 规则的基线，
    只有当政府债务/GDP 超过阈值时才会被规则上调，正常情况下基本保持配置值。
    想问"退休年龄如何影响经济"这类问题，rule_based 就足够可靠。
- type=central_bank（央行）。
  - base_interest_rate：rule_based 下由泰勒规则根据通胀/增长响应式计算，不直接采用配置的初始值，
    但会对真实的通胀/增长状态做出反应。
  - reserve_ratio：**rule_based 下硬编码为 0.08，忽略 YAML 配置**，仅当 gov_alg 属于 {llm, ppo, ddpg}
    时才真正生效。
- gov_alg=data_based 目前对政府角色有已知 bug（访问了不存在的属性），**禁止使用**。

## 3. market（市场/企业）——**默认选 perfect，除非问题本身关注市场竞争结构**

- type=perfect / monopoly：firm_n 必须为 1。
- type=oligopoly / monopolistic_competition：firm_n 可以 > 1，用户可以指定"几家企业"。
- 企业内部同质化：无法建模不同行业/产品异质性，只能通过 firm_n 控制"有几家竞争者"。

**选择规则（重要）**：只有当问题**本身**在问市场结构（垄断的福利损失、寡头合谋、产品多样性、
不同市场结构对比等）时才选非 perfect 类型；财政/货币/养老金/家庭行为类问题一律用 perfect。
原因：
1. firm_n > 1 会触发 env_core.expand_action_space——家庭动作维度 +=(1+firm_n)（选雇主 +
   对每家企业的消费分配），tax 政府动作维度 += firm_n（政府支出按企业分配）。
   monopolistic_competition 默认 firm_n=10，家庭动作会从 2 维涨到 13 维——RL 训练难度和
   所需训练量大幅增加，同样的 n_epochs 下更可能不收敛。
2. 非 perfect 时工资/价格不再由市场出清决定，而 rule_based 的企业定价是随机区间采样
   （如 monopolistic_competition 用 uniform(1.1,1.3) 乘生产率），会给结果引入大量与问题
   无关的噪声，污染政策对比实验。
3. perfect 是最小、最干净的基线：市场出清、无策略行为，问题焦点之外的一切都保持简单。

## 4. bank（银行）

- type=non_profit：统一利率，无决策空间。
- type=commercial：贷款/存款利率由算法决定；rule_based 下会真实响应央行基准利率、准备金率、
  贷款/存款总量，是可靠的杠杆。

## 5. 可用算法（house_alg / gov_alg / firm_alg / bank_alg）——如何按问题性质选择

这四个角色的"决策算法"不是越花哨越好，是四种不同的权衡，要按问题的性质挑：

- **rule_based**（基于经验规则/固定公式）：秒级出结果、确定性强，但只是"程式化的行为习惯"，
  不是真人决策，也不是最优决策；大量"政策类"参数（tau/xi/reserve_ratio/CRRA/IFE 等）在这个
  算法下不生效（见上）。**默认选它，除非问题明确需要下面几种更强的能力。**
- **bc**（行为克隆，household 默认算法）：基于真实调查数据（2022美国SCF）训练好的模型，
  零训练成本、速度和 rule_based 一样快，行为比 rule_based 的硬编码经验分布更贴近真实人类，
  适合"想要相对真实的家庭行为"的问题；但它同样不读 CRRA/IFE，不代表"最优"决策，仅代表
  "统计意义上真实的人类行为模式"。仅 households 可用。
- **ppo / ddpg / sac**（强化学习）：会根据 reward 函数（其中包含 CRRA/IFE/税率等参数）
  真正学习出**优化行为**，是让"政策类"参数产生真实行为差异、或者回答"最优决策应该是什么"
  这类问题的唯一途径。代价：需要训练（默认 300 epoch），真实规模场景（几百户、几百年）可能
  要几分钟到更久；训练量不够可能没收敛，效果比 rule_based 还不如；结果有随机性。
  **只在问题明确依赖"参数驱动的适应性/最优行为"时才选，并如实告知用户训练成本和收敛的不确定性。**
- **llm**（EconGym 自带的 llm_agent，仅支持 government / bank / market，不支持 households）：
  每一步都让一个 LLM 实时读取当前经济状态、做出决策，更接近"有推理能力的人类决策者"，
  不需要训练、能对参数即时响应。代价：每个 step 都是一次真实的 LLM API 调用，总调用次数
  ≈ episode_length × eval_episodes × config 数量，episode 越长、对比的 config 越多，
  耗时和成本越高；而且每次调用都有波动性，多次运行结果不完全一致。
  **适合"政府/银行/企业如果像一个会推理的决策者一样行事，会怎么做"这类问题**，
  不适合需要秒级出结果或大规模对比的场景。**硬性约束：episode_length × eval_episodes 不能
  超过 150**（校验层会拦截），选 llm 时务必把这两个值设小（比如 episode_length<=30,
  eval_episodes<=2），不要沿用 rule_based/bc 场景下 episode_length=300 这种默认值。
- **data_based**：仅 households 角色可靠（基于真实数据的启发式）；**政府角色有 bug，禁止使用**。

## 6. 输出指标（仿真结束后能拿到的真实数字）

GDP, per_gdp, income_gini, wealth_gini, social_welfare, house_wealth, house_income,
house_consumption, house_work_hours, house_age, gov_spending, house_total_tax,
retire_age, contribution_rate, pension_fund, dependency_ratio, inflation_rate,
base_interest_rate, reserve_ratio, deposit_rate, lending_rate,
gov_reward, house_reward, firm_reward, bank_reward, years。

注意：gov_reward / GDP / bank_reward / firm_reward 是"episode 内逐步求和"；
其余大多数指标（例如 tax_gov_reward / social_welfare）是"逐步取均值"。
两类字段量纲不同，解读时不能直接相加或类比。

**已知的数值边界情况**：income_gini / wealth_gini 理论上应该在 [0,1] 区间，但在经济被极端政策
（比如 llm/ppo government 设置了很激进的资产税）冲击到接近崩溃、财富分布出现大量负值/退化时，
EconGym 自身的 gini_coef() 计算可能输出**略微超出 [0,1] 甚至为负**的数值（这是 EconGym 自身
实现的数值边界情况，不是我们的 bug）。解读时如果看到 gini 值超出 [0,1] 或异常接近 0，
**不能**直接解读成"财富分配极度平等"，要结合 house_reward / social_welfare / house_wealth
是否也同时出现剧烈恶化，如果是，应说明"经济陷入了退化/崩溃状态，gini 数值在此时已经失去正常
的分配公平含义"。

## 7. 已知的建模边界（问题命中以下任意一条，应判定为不可行）

- 需要真实世界"尚未发生的未来数据"本身（如某官方机构的人口/GDP预测值）——应改写为"用真实美国数据
  校准起点，模拟 N 年后的趋势"这类可行任务，见第 0 节，不要直接判不可行。
- 涉及美国以外的其他具体国家的专门校准（没有其他国家的调查数据可用）。
- 涉及两个或以上经济体之间的贸易/关税/资本流动等互动（单一封闭经济体，无国际贸易机制）。
- 企业内部同质化，无法建模不同行业/产品异质性差异。
- 无离散冲击事件（战争、疫情、自然灾害等），除非能归约为现有参数的变化（如生产率骤降）。
- 家庭风险投资只有一个"股票市场"，无法建模房地产等其他资产类别。
- 无空间/地理维度，无法建模区域/城乡差异。
- 无法输出第 6 节列出的指标之外的任何数据（比如具体某个行业的产值、某个地区的失业率）。
"""
