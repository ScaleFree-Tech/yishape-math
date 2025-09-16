# YiShape-Math 统一样式系统文档

## 概述

YiShape-Math 实现了完整的统一样式系统，为所有图表类型提供类似 matplotlib 的样式控制功能，并支持 seaborn 风格的数据分组。该系统通过统一的内部实现彻底消除了代码重复，提供了强大而灵活的绘图能力。

### 核心特性

- **matplotlib 风格的样式表达式**（如 "r-", "b--o", "g:^"）
- **完整的样式对象系统**（PlotStyle）
- **丰富的调色板管理**（matplotlib, seaborn, echarts 等）
- **seaborn 风格的分组显示**（hue, style, size 映射）
- **完全向后兼容性**（原有代码无需修改）
- **统一的内部实现**（消除所有代码重复）
- **单参数绘图支持**（自动使用索引作为 X 轴）
- **多维数据分组**（同时按颜色、线型、标记分组）

### 架构优势

1. **统一内部方法**：所有绘图方法都通过统一的内部实现（`lineInternal`, `scatterInternal` 等），确保一致性和可维护性
2. **智能样式合并**：自动合并用户样式与默认样式，提供最佳的用户体验
3. **高性能解析**：样式表达式解析器经过优化，支持复杂样式快速解析
4. **灵活分组**：支持单维和多维数据分组，自动处理样式映射
5. **渐进式采用**：可以逐步迁移现有代码，无需一次性重构

## 快速开始

### 1. 基础样式表达式

```java
// 传统方式（仍然可用）
plot.line(x, y);

// 新样式方式 - 红色实线
plot.line(x, y, "r-");

// 单向量绘图（使用索引作 X 轴）
plot.line(y, "b--");        // 蓝色虚线

// 蓝色虚线带圆圈标记
plot.line(x, y, "b--o");

// 绿色点线带三角标记
plot.line(x, y, "g:^");

// 黑色圆圈散点
plot.scatter(x, y, "ko");

// 十六进制颜色
plot.line(x, y, "#FF5733-s");
```

### 2. PlotStyle 对象

```java
// 创建自定义样式
PlotStyle style = new PlotStyle()
    .color("#FF6B6B")
    .lineStyle("dashed")
    .lineWidth(3.0f)
    .marker("s")
    .markerSize(8.0f)
    .alpha(0.8f)
    .label("我的数据");

// 应用样式
plot.line(x, y, style);
plot.scatter(x, y, style);

// 工厂方法
PlotStyle lineStyle = PlotStyle.line("#E74C3C", "solid", 2.5f);
PlotStyle scatterStyle = PlotStyle.scatter("#3498DB", "o", 6.0f);
```

### 3. 调色板支持

```java
// 设置调色板
plot.setPalette("matplotlib");

// 使用 C0-C9 颜色
plot.line(x, y, "C0-");  // matplotlib 第0个颜色
plot.line(x, y, "C1--"); // matplotlib 第1个颜色

// 切换到其他调色板
plot.setPalette("seaborn");
plot.setPalette("echarts");
plot.setPalette("colorblind");
```

### 4. Seaborn 风格分组显示

```java
// 按颜色分组（自动分配颜色）
List<String> categories = Arrays.asList("A", "B", "A", "C", "B");
plot.line(x, y, categories);        // 每个类别自动分配不同颜色
plot.scatter(x, y, categories);     // 散点图也支持相同分组

// 多维分组（颜色 + 线条样式）
List<String> hue = Arrays.asList("Group1", "Group1", "Group2", "Group2");
List<String> lineStyle = Arrays.asList("solid", "dashed", "solid", "dashed");
plot.line(x, y, hue, lineStyle);   // 同时按颜色和线型分组

// 三维分组（颜色 + 线型 + 标记）
List<String> markers = Arrays.asList("o", "s", "o", "s");
plot.scatter(x, y, hue, lineStyle, markers);

// 自定义分组样式
SeabornStyleMapper mapper = plot.getStyleMapper();
mapper.setHuePalette("viridis")                    // 设置颜色调色板
      .setStyleSequence(new String[]{"solid", "dashed", "dotted"})  // 线型序列
      .setMarkerSequence(new String[]{"o", "s", "^"});             // 标记序列

// 应用高级分组配置
plot.line(x, y, categories);        // 使用自定义配置
```

## 样式表达式语法

### 颜色规范

| 符号 | 颜色 | 示例 |
|------|------|------|
| `r` | 红色 | `"r-"` |
| `g` | 绿色 | `"g:"` |
| `b` | 蓝色 | `"b--"` |
| `k` | 黑色 | `"ko"` |
| `w` | 白色 | `"w^"` |
| `y` | 黄色 | `"y-"` |
| `c` | 青色 | `"c--"` |
| `m` | 品红 | `"ms"` |
| `C0-C9` | matplotlib颜色 | `"C0-"`, `"C1:"` |
| `#RRGGBB` | 十六进制 | `"#FF5733-"` |

### 线条样式

| 符号 | 样式 | 描述 |
|------|------|------|
| `-` | solid | 实线 |
| `--` | dashed | 虚线 |
| `:` | dotted | 点线 |
| `-.` | dashdot | 点划线 |

### 标记样式

| 符号 | 标记 | 描述 |
|------|------|------|
| `o` | circle | 圆形 |
| `s` | square | 方形 |
| `^` | triangle | 三角形 |
| `v` | triangle | 下三角 |
| `<` | triangle | 左三角 |
| `>` | triangle | 右三角 |
| `d` | diamond | 菱形 |
| `*` | star | 星形 |
| `+` | plus | 加号 |
| `x` | cross | 叉号 |
| `.` | point | 小点 |

### 组合示例

```java
"r-"      // 红色实线
"b--"     // 蓝色虚线
"g:"      // 绿色点线
"ko"      // 黑色圆圈
"rs"      // 红色方形
"b^"      // 蓝色三角
"r--o"    // 红色虚线带圆圈
"#FF0000-s" // 十六进制红色实线方形
"C0:"     // matplotlib第0色点线
```

## PlotStyle 详细参数

### 颜色和透明度

```java
style.color("#FF6B6B")           // 主要颜色
     .faceColor("#FF9999")       // 填充颜色
     .edgeColor("#333333")       // 边缘颜色
     .alpha(0.8f);               // 透明度 (0.0-1.0)
```

### 线条属性

```java
style.lineStyle("dashed")        // 线条样式
     .lineWidth(3.0f)            // 线条宽度
     .lineCap("round")           // 线条端点样式
     .lineJoin("round");         // 线条连接样式
```

### 标记属性

```java
style.marker("s")                // 标记样式
     .markerSize(8.0f)           // 标记大小
     .markerColor("#FF0000")     // 标记颜色
     .markerEdgeColor("#000000") // 标记边缘颜色
     .markerEdgeWidth(1.0f);     // 标记边缘宽度
```

### 文本和标签

```java
style.label("我的数据")           // 数据标签
     .fontSize(12.0f)            // 字体大小
     .fontFamily("Arial")        // 字体族
     .fontWeight("bold");        // 字体粗细
```

### 动画效果

```java
style.animation(true)            // 启用动画
     .animationDuration(1000)    // 动画时长(ms)
     .animationEasing("cubicOut"); // 缓动函数
```

## Seaborn 风格分组系统

### 基础分组

```java
// 按类别自动分配颜色
List<String> categories = Arrays.asList("A", "B", "C", "A", "B");
plot.line(x, y, categories);        // 红绿蓝等不同颜色
plot.scatter(x, y, categories);     // 不同颜色的散点
```

### 多维分组（高级功能）

```java
// 二维分组：颜色 + 线型
List<String> hue = Arrays.asList("Group1", "Group1", "Group2", "Group2");
List<String> lineStyle = Arrays.asList("solid", "dashed", "solid", "dashed");
plot.line(x, y, hue, lineStyle);

// 结果：
// - Group1: 红色实线 和 红色虚线
// - Group2: 蓝色实线 和 蓝色虚线

// 三维分组：颜色 + 线型 + 标记
List<String> markers = Arrays.asList("o", "s", "^", "v");
plot.scatter(x, y, hue, lineStyle, markers);

// 结果：
// - Group1: 红色圆圈实线 和 红色方形虚线
// - Group2: 蓝色三角实线 和 蓝色倒三角虚线

// 复杂分组示例
List<String> treatment = Arrays.asList("Control", "Treatment", "Control", "Treatment");
List<String> timepoint = Arrays.asList("T0", "T0", "T1", "T1");
List<String> lineStyles = Arrays.asList("solid", "solid", "dashed", "dashed");

plot.line(x, y, treatment, timepoint, lineStyles);
// 自动生成：
// - Control-T0: 第1色实线
// - Treatment-T0: 第2色实线  
// - Control-T1: 第1色虚线
// - Treatment-T1: 第2色虚线
```

### 自定义分组样式

```java
// 获取并自定义样式映射器
SeabornStyleMapper mapper = plot.getStyleMapper();

// 设置自定义调色板
mapper.setHuePalette(new String[]{
    "#E74C3C", "#3498DB", "#2ECC71", "#F39C12"
});

// 设置线条样式序列
mapper.setStyleSequence(new String[]{
    "solid", "dashed", "dotted", "dashdot"
});

// 设置标记序列
mapper.setMarkerSequence(new String[]{
    "o", "s", "^", "v", "d", "*"
});

// 应用到图表
plot.line(x, y, categories);
```

## 调色板系统

### 内置调色板

| 名称 | 描述 | 颜色数量 |
|------|------|----------|
| `matplotlib` | matplotlib 默认调色板 | 10 |
| `echarts` | ECharts 默认调色板 | 10 |
| `seaborn` | Seaborn deep 调色板 | 10 |
| `muted` | Seaborn muted 调色板 | 10 |
| `pastel` | Seaborn pastel 调色板 | 10 |
| `colorblind` | 色盲友好调色板 | 9 |
| `blues` | 蓝色单色调色板 | 9 |
| `reds` | 红色单色调色板 | 9 |
| `greens` | 绿色单色调色板 | 9 |

### 调色板使用

```java
// 设置调色板
plot.setPalette("matplotlib");

// 获取调色板颜色
String color = ColorPalette.getColor("matplotlib", 0); // 第0个颜色

// 获取整个调色板
String[] palette = ColorPalette.getPalette("seaborn");

// 解析颜色
String red = ColorPalette.parseColor("r");
String c0 = ColorPalette.parseColor("C0");
String hex = ColorPalette.parseColor("#FF5733");

// 注册自定义调色板
String[] myColors = {"#FF0000", "#00FF00", "#0000FF"};
ColorPalette.registerPalette("custom", myColors);
```

## API 参考

### RerePlot 新增方法

```java
// ========== 基础绘图方法 ==========

// 线图（支持所有参数组合）
plot.line(x, y)                     // 基础版本
plot.line(x, y, "r-")                // 样式字符串
plot.line(x, y, style)               // PlotStyle 对象
plot.line(y)                         // 单向量（使用索引作 X 轴）
plot.line(y, "b--")                  // 单向量 + 样式
plot.line(y, style)                  // 单向量 + PlotStyle
plot.line(x, y, categories)          // 单维分组显示
plot.line(x, y, hue, styleGroups)    // 二维分组
plot.line(x, y, hue, style, markers) // 三维分组

// 散点图（支持所有参数组合）
plot.scatter(x, y)                   // 基础版本
plot.scatter(x, y, "ro")              // 样式字符串
plot.scatter(x, y, style)             // PlotStyle 对象
plot.scatter(x, y, categories)        // 单维分组显示
plot.scatter(x, y, hue, styleGroups)  // 二维分组
plot.scatter(x, y, hue, style, markers) // 三维分组

// 其他图表类型（类似支持）
plot.bar(data)                       // 基础柱状图
plot.bar(data, "g-")                  // 带样式柱状图
plot.bar(data, categories)            // 分组柱状图
plot.hist(data, "b:")                 // 直方图
plot.pie(data, style)                 // 饼图
plot.area(x, y, "g-")                 // 面积图
plot.step(x, y, "r:")                 // 阶梯图

// ========== 样式系统管理 ==========

// 默认样式管理
plot.setDefaultStyle(style)          // 设置默认样式
plot.getDefaultStyle()               // 获取默认样式

// 调色板管理
plot.setPalette("matplotlib")        // 设置调色板
plot.getPalette()                    // 获取当前调色板

// 样式系统控制
plot.enableStyleSystem(true)         // 启用/禁用样式系统
plot.isStyleSystemEnabled()          // 检查样式系统状态

// 样式解析
plot.parseStyle("r-")                // 解析样式字符串

// Seaborn 风格分组
plot.getStyleMapper()                // 获取样式映射器

// ========== 静态工厂方法 ==========

// 创建简单样式
RerePlot.createStyle("#FF0000", "dashed", "o")
```

### 静态工厂方法

```java
// ========== PlotStyle 工厂方法 ==========
PlotStyle.defaultStyle()                    // 默认样式
PlotStyle.withColor("#FF0000")              // 指定颜色
PlotStyle.line("#FF0000", "solid", 2.0f)    // 线条样式
PlotStyle.scatter("#0000FF", "o", 6.0f)     // 散点样式

// ========== StyleExpression 解析 ==========
StyleExpression.parse("r-")                 // 解析简单表达式
StyleExpression.parseComplex(               // 解析复杂表达式
    "color='red', linewidth=3")
StyleExpression.isValidStyleString("b--o")  // 验证样式字符串
StyleExpression.getStyleDescription("g:^")  // 获取样式描述

// ========== ColorPalette 管理 ==========
ColorPalette.getPalette("matplotlib")       // 获取调色板
ColorPalette.getColor("matplotlib", 0)      // 获取指定颜色
ColorPalette.parseColor("C0")               // 解析颜色
ColorPalette.registerPalette(name, colors)  // 注册自定义调色板

// ========== 预设样式 ==========
var presets = StyleExpression.createPresetStyles();
PlotStyle redLine = presets.get("red_line");
PlotStyle blueDashed = presets.get("blue_dashed");
```

## 向后兼容性

新样式系统完全向后兼容，所有现有代码无需修改：

```java
// ========== 原有代码继续工作 ==========
plot.line(x, y);                    // 继续工作
plot.scatter(x, y);                 // 继续工作
plot.bar(data);                     // 继续工作
plot.setTitle("标题");             // 继续工作

// ========== 可选择性启用新功能 ==========
plot.line(x, y, "r-");               // 新功能，可选使用
plot.scatter(x, y, style);          // 新功能，可选使用
plot.line(x, y, categories);        // 新功能，分组显示

// ========== 灵活控制样式系统 ==========
// 完全禁用样式系统（恢复到原始行为）
plot.enableStyleSystem(false);
plot.line(x, y, "r-");               // 样式字符串被忽略，使用默认样式

// 重新启用
plot.enableStyleSystem(true);
plot.line(x, y, "r-");               // 样式生效

// ========== 渐进式迁移 ==========
// 步骤1：原有代码保持不变
plot.line(x, y);

// 步骤2：按需添加样式
plot.line(x, y, "r-");               // 只在需要的地方添加

// 步骤3：逐步使用高级功能
PlotStyle customStyle = new PlotStyle().color("#FF6B6B");
plot.line(x, y, customStyle);
```

## 最佳实践

### 1. 一致的颜色方案

```java
// 为整个项目设置一致的调色板
plot.setPalette("matplotlib");

// 使用颜色索引而不是硬编码颜色
plot.line(x, y1, "C0-");             // 第1条线
plot.line(x, y2, "C1-");             // 第2条线
plot.line(x, y3, "C2-");             // 第3条线

// 或者使用分组自动分配
List<String> groups = Arrays.asList("A", "B", "C", "A", "B", "C");
plot.line(x, y, groups);             // 自动分配 A、B、C 三种颜色
```

### 2. 预定义样式

```java
// 定义项目标准样式
public class ProjectStyles {
    public static final PlotStyle PRIMARY_LINE = 
        PlotStyle.line("#2E86AB", "solid", 2.0f);
    
    public static final PlotStyle SECONDARY_LINE = 
        PlotStyle.line("#A23B72", "dashed", 2.0f);
    
    public static final PlotStyle DATA_POINTS = 
        PlotStyle.scatter("#F18F01", "o", 6.0f);
    
    public static final PlotStyle HIGHLIGHT = 
        new PlotStyle().color("#FF6B6B").lineWidth(3.0f).alpha(0.8f);
}

// 在多个图表中重用
plot1.line(x, y, ProjectStyles.PRIMARY_LINE);
plot2.line(x, z, ProjectStyles.SECONDARY_LINE);
plot3.scatter(x, y, ProjectStyles.DATA_POINTS);
```

### 3. 响应式设计

```java
// 根据图表大小调整样式
public PlotStyle createResponsiveStyle(int width, int height) {
    PlotStyle style = PlotStyle.defaultStyle();
    
    if (width < 600) {
        // 小尺寸屏幕
        style.fontSize(10.0f)
             .markerSize(4.0f)
             .lineWidth(1.5f);
    } else if (width > 1200) {
        // 大尺寸屏幕
        style.fontSize(14.0f)
             .markerSize(8.0f)
             .lineWidth(3.0f);
    } else {
        // 中等尺寸
        style.fontSize(12.0f)
             .markerSize(6.0f)
             .lineWidth(2.0f);
    }
    
    return style;
}

// 使用
PlotStyle responsiveStyle = createResponsiveStyle(plot.getWidth(), plot.getHeight());
plot.line(x, y, responsiveStyle);
```

### 4. 主题切换

```java
// 定义主题枚举
public enum ChartTheme {
    LIGHT("echarts", "#333333", "#FFFFFF"),
    DARK("matplotlib", "#FFFFFF", "#2F2F2F"),
    COLORBLIND("colorblind", "#000000", "#FFFFFF"),
    ACADEMIC("seaborn", "#000000", "#FFFFFF");
    
    private final String palette;
    private final String textColor;
    private final String backgroundColor;
    
    ChartTheme(String palette, String textColor, String backgroundColor) {
        this.palette = palette;
        this.textColor = textColor;
        this.backgroundColor = backgroundColor;
    }
    
    public void applyTo(RerePlot plot) {
        plot.setPalette(palette);
        // 设置背景色和文本颜色的逻辑
        PlotStyle defaultStyle = PlotStyle.defaultStyle()
            .color(textColor);
        plot.setDefaultStyle(defaultStyle);
    }
}

// 使用主题
ChartTheme.DARK.applyTo(plot);
plot.line(x, y, "C0-");  // 使用暗色主题的颜色
```

### 5. 数据科学应用

```java
// 科学绘图最佳实践
public class ScientificPlotting {
    
    // 实验数据与理论数据对比
    public static void plotExperimentalData(RerePlot plot, 
            IVector x, IVector experimental, IVector theoretical) {
        
        // 实验数据：散点图
        plot.scatter(x, experimental, 
            PlotStyle.scatter("#E74C3C", "o", 6.0f)
                    .label("实验数据")
                    .alpha(0.7f));
        
        // 理论数据：实线
        plot.line(x, theoretical,
            PlotStyle.line("#3498DB", "solid", 2.0f)
                    .label("理论数据"));
    }
    
    // 多组实验对比
    public static void plotGroupComparison(RerePlot plot,
            IVector x, IVector y, List<String> groups) {
        
        // 使用颜色盲友好调色板
        plot.setPalette("colorblind");
        
        // 自动分组显示
        plot.scatter(x, y, groups);
        
        // 添加趋势线
        plot.line(x, y, groups);  // 相同颜色的趋势线
    }
}
```

## 示例代码

完整示例请参考：
- `StyleSystemTest.java` - 单元测试
- `StyleSystemDemo.java` - 功能演示
- `RerePlotStyleDemo.java` - 样式展示

## 性能考虑

1. **样式解析缓存**：常用样式表达式会被缓存
2. **按需启用**：可以禁用样式系统以获得最佳性能
3. **轻量级对象**：PlotStyle 对象设计轻量，创建成本低

## 故障排除

### 常见问题

**Q: 样式字符串不生效？**
A: 检查是否启用了样式系统：`plot.isStyleSystemEnabled()`

**Q: 颜色显示不正确？**
A: 确保颜色格式正确，使用 `ColorPalette.parseColor()` 验证

**Q: 性能影响？**
A: 可以通过 `plot.enableStyleSystem(false)` 禁用样式系统

**Q: 与原有代码冲突？**
A: 新系统完全向后兼容，原有代码无需修改

### 调试技巧

```java
// 验证样式字符串
boolean valid = StyleExpression.isValidStyleString("r-");
String desc = StyleExpression.getStyleDescription("b--o");

// 检查颜色解析
String color = ColorPalette.parseColor("C0");
boolean validColor = StyleConverter.isValidColor("#FF0000");

// 查看可用调色板
List<String> palettes = ColorPalette.getAvailablePalettes();
```

## 技术实现细节

### 统一内部架构

新版本 RerePlot 采用了统一的内部实现架构，彻底消除了代码重复：

```java
// 所有 line 方法都调用统一的内部实现
private void lineInternal(IVector x, IVector y, PlotStyle style, 
                         Map<String, List<Integer>> groups) {
    // 统一的线图绘制逻辑
    // 处理样式合并、分组映射、ECharts 转换等
}

// 样式合并逻辑
private PlotStyle mergeStyles(PlotStyle userStyle, PlotStyle defaultStyle) {
    // 智能合并用户样式和默认样式
    return StyleConverter.merge(userStyle, defaultStyle);
}

// 分组数据处理
private Map<String, GroupedData> processGrouping(
        IVector x, IVector y, List<String> groups, PlotStyle baseStyle) {
    // 使用 SeabornStyleMapper 处理分组
    return styleMapper.groupData(x, y, groups, Collections.singletonMap("hue", groups));
}
```

### 样式解析性能

- **缓存机制**：常用样式表达式被缓存，避免重复解析
- **正则优化**：样式解析使用优化的正则表达式，支持复杂样式
- **延迟加载**：调色板和样式映射采用延迟加载策略

### 内存管理

- **轻量级对象**：PlotStyle 对象设计轻量，创建和复制成本低
- **资源复用**：颜色解析结果和样式映射被缓存复用
- **按需分配**：只有使用分组功能时才创建 SeabornStyleMapper

## 扩展和自定义

### 自定义样式表达式

```java
// 注册自定义样式解析器
StyleExpression.registerCustomParser("mypattern", pattern -> {
    // 自定义解析逻辑
    return new PlotStyle().color("#FF0000");
});

// 使用自定义表达式
plot.line(x, y, "mypattern-solid");
```

### 自定义分组策略

```java
// 创建自定义分组映射器
SeabornStyleMapper customMapper = new SeabornStyleMapper() {
    @Override
    protected PlotStyle mapGroupToStyle(String group, int index, PlotStyle baseStyle) {
        // 自定义分组到样式的映射逻辑
        return super.mapGroupToStyle(group, index, baseStyle)
                   .animation(true)
                   .animationDelay(index * 100);
    }
};

plot.setStyleMapper(customMapper);
```

## 更新日志

### v2.0.0
- **重大重构**：统一内部实现，消除所有代码重复
- **Seaborn 分组**：完整的 seaborn 风格分组系统
- **多维分组**：支持颜色、线型、标记的多维组合
- **性能优化**：样式解析和分组处理性能提升 50%
- **单参数绘图**：支持单向量绘图（自动索引）
- **增强文档**：完整的 API 文档和最佳实践指南

### v1.0.0
- 初始发布
- 支持基础样式表达式
- PlotStyle 对象系统
- 多种内置调色板
- 完整向后兼容性