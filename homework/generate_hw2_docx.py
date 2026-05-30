#!/usr/bin/env python3
"""
生成第二次作业 Word 文档
科学计算与机器学习 - Allen-Cahn方程PINN求解
"""

import json
import os
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# 设置中文字体
def set_chinese_font(run, font_name='宋体'):
    """设置中文字体"""
    run.font.name = font_name
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font_name)

# 设置段落格式
def set_paragraph_style(paragraph, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY, first_line_indent=True):
    """设置段落格式"""
    paragraph.alignment = alignment
    if first_line_indent:
        paragraph.paragraph_format.first_line_indent = Inches(0.3)
    paragraph.paragraph_format.line_spacing = 1.5
    paragraph.paragraph_format.space_after = Pt(6)

# 添加标题
def add_heading(doc, text, level=1):
    """添加标题"""
    heading = doc.add_heading(text, level=level)
    for run in heading.runs:
        set_chinese_font(run)
        run.font.size = Pt(16 if level == 1 else 14)
    return heading

# 添加正文段落
def add_paragraph(doc, text, first_line_indent=True):
    """添加正文段落"""
    para = doc.add_paragraph(text)
    set_paragraph_style(para, first_line_indent=first_line_indent)
    for run in para.runs:
        set_chinese_font(run)
        run.font.size = Pt(12)
    return para

# 添加图片
def add_picture(doc, picture_path, width=Inches(5)):
    """添加图片"""
    if os.path.exists(picture_path):
        paragraph = doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run()
        run.add_picture(picture_path, width=width)
        set_paragraph_style(paragraph, first_line_indent=False)
        return paragraph
    else:
        add_paragraph(doc, f"[图片不存在: {os.path.basename(picture_path)}]")
        return None

# 添加图片标题
def add_picture_caption(doc, text):
    """添加图片标题"""
    paragraph = doc.add_paragraph(text)
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    set_paragraph_style(paragraph, first_line_indent=False)
    for run in paragraph.runs:
        set_chinese_font(run)
        run.font.size = Pt(10.5)
    return paragraph

# 创建表格
def create_table(doc, rows, cols, data):
    """创建表格并填充数据"""
    table = doc.add_table(rows=rows, cols=cols)
    table.style = 'Light Grid Accent 1'

    for i in range(rows):
        for j in range(cols):
            cell = table.rows[i].cells[j]
            cell.text = str(data[i][j])
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    set_chinese_font(run)
                    run.font.size = Pt(10.5)
            # 设置对齐
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
                if i == 0:  # 表头加粗
                    for run in paragraph.runs:
                        run.font.bold = True

    return table

# 读取实验结果
def load_experiment_results():
    """读取所有实验结果"""
    base_dir = "/home/xuanli/work/SciML-Course/homework/allen_cahn_pinn/outputs"
    experiments = {
        'uniform_equal': '均匀采样 + 等权重',
        'random_equal': '随机采样 + 等权重',
        'random_adaptive_points': '随机采样 + 自适应加点',
        'random_adaptive_weights': '随机采样 + 自适应权重'
    }

    results = {}
    for exp_key, exp_name in experiments.items():
        metrics_path = os.path.join(base_dir, exp_key, 'metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                results[exp_key] = {
                    'name': exp_name,
                    'metrics': json.load(f),
                    'figures_dir': os.path.join(base_dir, exp_key, 'figures')
                }
    return results

def main():
    # 创建文档
    doc = Document()

    # 设置默认字体
    doc.styles['Normal'].font.name = '宋体'
    doc.styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')
    doc.styles['Normal'].font.size = Pt(12)

    # ========== 标题部分 ==========
    title = doc.add_heading('科学计算第二次作业：Allen-Cahn方程的PINN求解', level=0)
    for run in title.runs:
        set_chinese_font(run)
        run.font.size = Pt(18)
        run.font.bold = True

    # 作者信息
    info_para = doc.add_paragraph()
    info_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    info_run = info_para.add_run('D202581671  人工智能与自动化学院  李玄')
    set_chinese_font(info_run)
    info_run.font.size = Pt(12)
    doc.add_paragraph()  # 空行

    # ========== 摘要 ==========
    add_heading(doc, '摘要', level=1)
    abstract_text = (
        '本报告针对Allen-Cahn方程的一维初边值问题，采用物理信息神经网络(PINN)方法进行数值求解。'
        'Allen-Cahn方程是一类重要的反应扩散方程，广泛应用于相场模型、图像处理等领域。'
        '实验采用连续时间PINN形式，用神经网络逼近解u(x,t)，通过自动微分计算PDE残差，'
        '构造包含PDE残差项、初值项和边界项的联合损失函数进行训练。'
        '本作业系统考察了采样策略（均匀采样 vs 随机采样）、自适应加点策略和自适应权重策略对求解精度的影响。'
        '实验结果表明：均匀采样在本问题中表现最佳，PDE残差MSE达到2.061×10⁻²；'
        '自适应加点策略能有效降低PDE残差，将MSE降至1.335×10⁻²；'
        '自适应权重策略在各项损失间取得较好平衡。'
        '通过可视化分析，PINN成功捕获了Allen-Cahn方程的解的时空演化特征，'
        '在不同时刻的截面曲线均呈现出合理的物理行为。'
        '关键词：物理信息神经网络；Allen-Cahn方程；自适应采样；损失权重；反应扩散方程。'
    )
    add_paragraph(doc, abstract_text, first_line_indent=False)
    doc.add_paragraph()  # 空行

    # ========== 第一章 ==========
    add_heading(doc, '1  作业背景与研究目标', level=1)

    add_paragraph(doc,
        'Allen-Cahn方程是一类重要的非线性反应扩散方程，其一般形式为u_t = D·u_xx + f(u)，'
        '其中f(u)通常为双稳势函数的导数。该方程在相场模型、材料科学、图像处理等领域有广泛应用。'
        '本作业要求使用PyTorch实现一维Allen-Cahn方程的PINN求解器，并考察不同训练策略对求解精度的影响。')
    doc.add_paragraph()

    add_paragraph(doc,
        '具体而言，本作业围绕以下三个研究问题展开：其一，均匀采样与随机采样哪一种配点策略更适合本问题；'
        '其二，自适应加点策略是否能有效提升PDE残差的收敛速度；'
        '其三，自适应权重策略能否平衡不同损失项，提升整体求解精度。')
    doc.add_paragraph()

    # ========== 第二章 ==========
    add_heading(doc, '2  问题描述与PINN方法', level=1)

    add_heading(doc, '2.1  Allen-Cahn方程问题定义', level=2)
    add_paragraph(doc,
        '本报告求解的一维Allen-Cahn方程初边值问题定义如下：')
    doc.add_paragraph()

    # 数学公式段落
    equation_para = doc.add_paragraph()
    equation_run = equation_para.add_run(
        'PDE: u_t - D·u_xx + 5(u³ - u) = 0,  D = 10⁻⁴\n'
        '求解域: x ∈ [-1, 1],  t ∈ [0, 1]\n'
        '初值条件: u(x, 0) = x²cos(πx)\n'
        '边界条件: u(-1, t) = u(1, t) = -1'
    )
    set_chinese_font(equation_run)
    equation_run.font.size = Pt(11)
    doc.add_paragraph()

    add_heading(doc, '2.2  PINN方法原理', level=2)
    add_paragraph(doc,
        '物理信息神经网络(PINN)的核心思想是将PDE作为正则化项嵌入神经网络的损失函数中。')
    add_paragraph(doc,
        '设神经网络为u_θ(x,t)，通过自动微分计算u_t、u_x、u_xx，构造PDE残差：')
    doc.add_paragraph()

    residual_para = doc.add_paragraph()
    residual_run = residual_para.add_run(
        'f(x,t) = u_t - D·u_xx + 5(u³ - u)'
    )
    set_chinese_font(residual_run)
    residual_run.font.size = Pt(11)
    doc.add_paragraph()

    add_paragraph(doc,
        '总损失函数由三部分组成：')
    doc.add_paragraph()

    loss_para = doc.add_paragraph()
    loss_run = loss_para.add_run(
        'L = λ_f·L_f + λ_ic·L_ic + λ_bc·L_bc\n'
        '其中:\n'
        '  L_f = MSE[f(x_f, t_f)]  (PDE残差)\n'
        '  L_ic = MSE[u(x_ic, 0) - u₀(x_ic)]  (初值条件)\n'
        '  L_bc = MSE[u(-1, t_bc) + 1] + MSE[u(1, t_bc) + 1]  (边界条件)'
    )
    set_chinese_font(loss_run)
    loss_run.font.size = Pt(11)
    doc.add_paragraph()

    # ========== 第三章 ==========
    add_heading(doc, '3  实验设计与实现', level=1)

    add_heading(doc, '3.1  网络结构与训练设置', level=2)
    add_paragraph(doc,
        '实验采用PyTorch框架实现，神经网络结构为多层感知机(MLP)。'
        '默认配置使用4层隐藏层，每层64个神经元，激活函数采用tanh。'
        '优化器使用Adam，学习率为0.001，训练轮数为1500 epoch。')
    add_paragraph(doc,
        '配点设置方面，初始内部配点数为2000，初值配点数为100，边界配点数为100。'
        '对于自适应加点策略，在训练1000 epoch后，在候选点上计算残差，'
        '将残差最大的800个点加入配点集继续训练。')

    add_heading(doc, '3.2  实验方案', level=2)
    add_paragraph(doc, '本报告设计了四组对比实验，具体设置如表1所示：')
    doc.add_paragraph()

    # 实验方案表格
    table_data = [
        ['实验组', '采样方式', '权重策略', '自适应加点'],
        ['实验一', '均匀采样', '等权重', '否'],
        ['实验二', '随机采样', '等权重', '否'],
        ['实验三', '随机采样', '等权重', '是'],
        ['实验四', '随机采样', '自适应权重', '否']
    ]
    create_table(doc, 5, 4, table_data)
    doc.add_paragraph()

    # ========== 第四章 ==========
    add_heading(doc, '4  实验结果与分析', level=1)

    # 加载实验结果
    results = load_experiment_results()

    # 4.1 整体性能对比
    add_heading(doc, '4.1  整体性能对比', level=2)
    add_paragraph(doc, '表2给出了四组实验的定量指标对比：')
    doc.add_paragraph()

    # 结果对比表格
    summary_data = [['实验组', 'PDE MSE', 'IC MSE', 'BC MSE', '相对L2误差', '训练时间(s)']]
    for exp_key in ['uniform_equal', 'random_equal', 'random_adaptive_points', 'random_adaptive_weights']:
        if exp_key in results:
            m = results[exp_key]['metrics']
            summary_data.append([
                results[exp_key]['name'],
                f"{m['pde_residual_mse']:.4e}",
                f"{m['ic_mse']:.4e}",
                f"{m['bc_mse']:.4e}",
                f"{m['relative_l2_error']:.4f}",
                f"{m['elapsed_sec']:.1f}"
            ])
    create_table(doc, 5, 6, summary_data)
    doc.add_paragraph()

    add_paragraph(doc,
        '从表2可以看出：')
    add_paragraph(doc,
        '(1) 均匀采样策略在本问题中表现最优，PDE残差MSE为2.061×10⁻²，'
        '明显优于随机采样的3.330×10⁻²。这说明对于Allen-Cahn方程这类具有平滑解结构的问题，'
        '均匀覆盖求解域能够更有效地学习解的时空演化特征。')
    add_paragraph(doc,
        '(2) 自适应加点策略显著降低了PDE残差，MSE降至1.335×10⁻²，为四组实验中最低。'
        '这表明通过在残差较大的区域增加配点密度，可以有效提升PINN对PDE的满足程度。')
    add_paragraph(doc,
        '(3) 自适应权重策略在各项损失间取得了较好的平衡，'
        'PDE残差MSE为2.224×10⁻²，介于均匀采样和随机采样之间。')

    # 4.2 实验一：均匀采样+等权重
    add_heading(doc, '4.2  实验一：均匀采样 + 等权重', level=2)
    exp1 = results.get('uniform_equal')
    if exp1:
        add_paragraph(doc,
            f'该实验采用均匀网格采样配点，所有损失权重设为1。'
            f'训练后PDE残差MSE为{exp1["metrics"]["pde_residual_mse"]:.4e}，'
            f'初值MSE为{exp1["metrics"]["ic_mse"]:.4e}，'
            f'边界MSE为{exp1["metrics"]["bc_mse"]:.4e}。')
        doc.add_paragraph()

        # 添加图片
        figures_dir = exp1['figures_dir']
        add_picture(doc, os.path.join(figures_dir, 'loss_curves.png'))
        add_picture_caption(doc, '图1  实验一：损失函数变化曲线')
        doc.add_paragraph()

        add_picture(doc, os.path.join(figures_dir, 'solution_heatmap.png'))
        add_picture_caption(doc, '图2  实验一：解的二维热力图')
        doc.add_paragraph()

        add_picture(doc, os.path.join(figures_dir, 'solution_slices.png'))
        add_picture_caption(doc, '图3  实验一：不同时刻的截面曲线')
        doc.add_paragraph()

        add_paragraph(doc,
            '从图1可以看到，总损失和各项子损失均呈现快速下降趋势。'
            '图2展示了u(x,t)在时空域上的整体演化，可以看到解从初始时刻的x²cos(πx)形状'
            '逐渐演化，边界条件得到满足。图3给出了t=0, 0.25, 0.5, 0.75, 1.0五个时刻的'
            '截面曲线，显示了方程解随时间的演化过程。')

    # 4.3 实验二：随机采样+等权重
    add_heading(doc, '4.3  实验二：随机采样 + 等权重', level=2)
    exp2 = results.get('random_equal')
    if exp2:
        add_paragraph(doc,
            f'该实验采用随机采样配点，所有损失权重设为1。'
            f'训练后PDE残差MSE为{exp2["metrics"]["pde_residual_mse"]:.4e}，'
            f'高于均匀采样实验。这表明随机采样在某些区域可能配点稀疏，'
            '导致局部PDE残差较大。')
        doc.add_paragraph()

        figures_dir = exp2['figures_dir']
        add_picture(doc, os.path.join(figures_dir, 'loss_curves.png'))
        add_picture_caption(doc, '图4  实验二：损失函数变化曲线')
        doc.add_paragraph()

        add_picture(doc, os.path.join(figures_dir, 'solution_heatmap.png'))
        add_picture_caption(doc, '图5  实验二：解的二维热力图')
        doc.add_paragraph()

        add_picture(doc, os.path.join(figures_dir, 'solution_slices.png'))
        add_picture_caption(doc, '图6  实验二：不同时刻的截面曲线')
        doc.add_paragraph()

    # 4.4 实验三：随机采样+自适应加点
    add_heading(doc, '4.4  实验三：随机采样 + 自适应加点', level=2)
    exp3 = results.get('random_adaptive_points')
    if exp3:
        add_paragraph(doc,
            f'该实验在随机采样基础上，采用自适应加点策略。'
            f'训练1000 epoch后，选择残差最大的800个点加入配点集继续训练。'
            f'最终PDE残差MSE为{exp3["metrics"]["pde_residual_mse"]:.4e}，'
            f'为四组实验中最低，证明自适应加点策略的有效性。')
        doc.add_paragraph()

        figures_dir = exp3['figures_dir']
        add_picture(doc, os.path.join(figures_dir, 'adaptive_points.png'))
        add_picture_caption(doc, '图7  实验三：自适应加点前后配点分布')
        doc.add_paragraph()

        add_picture(doc, os.path.join(figures_dir, 'loss_curves.png'))
        add_picture_caption(doc, '图8  实验三：损失函数变化曲线')
        doc.add_paragraph()

        add_picture(doc, os.path.join(figures_dir, 'solution_heatmap.png'))
        add_picture_caption(doc, '图9  实验三：解的二维热力图')
        doc.add_paragraph()

        add_picture(doc, os.path.join(figures_dir, 'solution_slices.png'))
        add_picture_caption(doc, '图10  实验三：不同时刻的截面曲线')
        doc.add_paragraph()

        add_paragraph(doc,
            '从图7可以清晰看到，自适应加点策略在残差较大的区域（主要是初始时刻和边界附近）'
            '增加了配点密度。这种有针对性的加点策略显著提升了PDE残差的收敛速度和精度。')

    # 4.5 实验四：随机采样+自适应权重
    add_heading(doc, '4.5  实验四：随机采样 + 自适应权重', level=2)
    exp4 = results.get('random_adaptive_weights')
    if exp4:
        add_paragraph(doc,
            f'该实验采用自适应权重策略，根据各项损失的相对大小动态更新权重系数。'
            f'最终PDE残差MSE为{exp4["metrics"]["pde_residual_mse"]:.4e}，'
            f'初值MSE为{exp4["metrics"]["ic_mse"]:.4e}，'
            f'边界MSE为{exp4["metrics"]["bc_mse"]:.4e}。')
        doc.add_paragraph()

        figures_dir = exp4['figures_dir']
        add_picture(doc, os.path.join(figures_dir, 'loss_curves.png'))
        add_picture_caption(doc, '图11  实验四：损失函数变化曲线（含自适应权重）')
        doc.add_paragraph()

        add_picture(doc, os.path.join(figures_dir, 'solution_heatmap.png'))
        add_picture_caption(doc, '图12  实验四：解的二维热力图')
        doc.add_paragraph()

        add_picture(doc, os.path.join(figures_dir, 'solution_slices.png'))
        add_picture_caption(doc, '图13  实验四：不同时刻的截面曲线')
        doc.add_paragraph()

    # ========== 第五章 ==========
    add_heading(doc, '5  结论', level=1)

    add_paragraph(doc,
        '本报告针对一维Allen-Cahn方程初边值问题，使用物理信息神经网络(PINN)方法进行数值求解，'
        '系统考察了采样策略、自适应加点和自适应权重策略对求解精度的影响，主要结论如下：')
    doc.add_paragraph()

    add_paragraph(doc,
        '(1) 采样策略方面：均匀采样在本问题中表现优于随机采样，'
        '这可能是因为Allen-Cahn方程的解具有较平滑的时空结构，均匀覆盖求解域更有利于学习。')
    add_paragraph(doc,
        '(2) 自适应加点策略：显著降低了PDE残差，证明了基于残差的自适应配点策略的有效性。'
        '该策略能够自动识别残差较大的区域并增加配点密度，提升局部精度。')
    add_paragraph(doc,
        '(3) 自适应权重策略：在各项损失间取得较好平衡，避免了某一损失项主导训练过程的问题。')
    add_paragraph(doc,
        '(4) PINN方法成功捕获了Allen-Cahn方程解的时空演化特征，'
        '在满足初边值条件的同时，较好地满足PDE约束。')
    doc.add_paragraph()

    add_paragraph(doc,
        '本作业的实验结果表明，PINN方法能够有效求解Allen-Cahn方程这类反应扩散问题，'
        '而合理的采样策略和自适应机制能够显著提升求解精度。'
        '未来工作可进一步探索网络结构优化、多尺度时间方法等方向。')

    # 保存文档
    output_path = "/home/xuanli/work/SciML-Course/homework/科学计算第二次作业.docx"
    doc.save(output_path)
    print(f"Word文档已生成: {output_path}")

if __name__ == '__main__':
    main()
