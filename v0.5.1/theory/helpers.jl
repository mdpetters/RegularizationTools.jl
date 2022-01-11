using Cairo 
using Fontconfig
using Gadfly 
using DataFrames 
using Colors
using LinearAlgebra 
using Underscores 
using Printf 
using NumericIO

function graph(df::DataFrame; colors = ["black", "darkred", "steelblue3", "darkgoldenrod"], autoscale = false)
    p1 = plot(
        df,
        x = :x,
        y = :y,
        color = :Color,
        Geom.line,
        Guide.xlabel("n"),
        Guide.ylabel(""),
        Scale.color_discrete_manual(colors...),
        Guide.colorkey(; title = ""),
        Theme(plot_padding = [1Gadfly.mm,5Gadfly.mm,1Gadfly.mm,1Gadfly.mm]),
    )
end

function graph1(residual, solution)
    mλ = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]
    mL1, mL2 = L1.(mλ), L2.(mλ)
    label = map(x->@sprintf("λ = %.0e",x), mλ)
    df = DataFrame(x = residual, y = solution)

    gengrid(r) = [vcat(map(x -> x:x:9x, r)...); r[end] * 10]
    xg = [log10.([0.8, 0.9]); log10.(gengrid([1])); log10.([20,30,40])]
    yg = [log10.([0.5, 0.6, 0.7, 0.8, 0.9]); log10.(gengrid([1, 10])); log10.([200])]
    xlabels = [-1,0,1]
    ylabels = [0,1,2,3]
    @_ lfuny = ifelse(sum(_ .== ylabels) == 1, formatted(exp10(_), :SI, ndigits = 1), "")
    @_ lfunx = ifelse(sum(_ .== ylabels) == 1, formatted(exp10(_), :SI, ndigits = 1), "")
    p = plot(df, x = :x, y = :y, Geom.line,
        layer(x = mL1, y = mL2, label = label, Geom.point, Geom.label),
        Scale.x_log10(labels = lfunx), 
        Scale.y_log10(labels = lfuny),
        Guide.xlabel("Residual norm ||𝐀*x-b||"),
        Guide.ylabel("Solution norm ||𝐋*(x-x<sub>0</sub>)||", orientation = :vertical),
        Guide.xticks(ticks = xg),
        Guide.yticks(ticks = yg),
        Theme(default_color="black"),
        Coord.cartesian(xmin = log10(8e-1), xmax = log10(40), ymin = log10(0.5), ymax = log10(200)))

end

function graph2(λs, V)
    gengrid(r) = [vcat(map(x -> x:x:9x, r)...); r[end] * 10]
    xlabels = log10.([0.001, 0.01, 0.1, 1])
    @_ lfunx = ifelse(sum(_ .== xlabels) == 1, @sprintf("%.3f", exp10(_)), "")

    plot(x = λs, y = V, Geom.line,  
        Theme(default_color="black"),
        Scale.x_log10(labels = lfunx), 
        Guide.xticks(ticks = log10.(gengrid([0.001, 0.01,0.1]))),
        Guide.ylabel("V(λ)"), 
        Guide.xlabel(" λ ")
        )

    
end

function graph4(residual, solution)
    df = DataFrame(x = residual, y = solution)

    gengrid(r) = [vcat(map(x -> x:x:9x, r)...); r[end] * 10]
    xg = [log10.([0.8, 0.9]); log10.(gengrid([1])); log10.([20,30,40])]
    yg = [log10.([0.05, 0.06, 0.07, 0.08, 0.09]); log10.(gengrid([0.1, 1])); log10.([20])]
    xlabels = [-1,0,1]
    ylabels = [-1,0,1]
    @_ lfuny = ifelse(sum(_ .== ylabels) == 1, formatted(exp10(_), :SI, ndigits = 1), "")
    @_ lfunx = ifelse(sum(_ .== ylabels) == 1, formatted(exp10(_), :SI, ndigits = 1), "")
    set_default_plot_size(10cm, 8cm) 

    p = plot(df, x = :x, y = :y, Geom.line,
        Scale.x_log10(labels = lfunx), 
        Scale.y_log10(labels = lfuny),
        Guide.xlabel("Residual norm ||𝐀*x-b||"),
        Guide.ylabel("Solution norm ||𝐋*(x-x<sub>0</sub>)||", orientation = :vertical),
        Guide.xticks(ticks = xg),
        Guide.yticks(ticks = yg),
        Theme(default_color="black"),
        Coord.cartesian(xmin = log10(8e-1), xmax = log10(40), ymin = log10(0.05), ymax = log10(20)))

end


function graph3(λs, V)
    gengrid(r) = [vcat(map(x -> x:x:9x, r)...); r[end] * 10]
    xlabels = log10.([0.1, 1, 10])
    @_ lfunx = ifelse(sum(_ .== xlabels) == 1, @sprintf("%.1f", exp10(_)), "")
    set_default_plot_size(10cm, 8cm) 
    plot(x = λs, y = V, Geom.line,  
        Theme(default_color="black"),
        Scale.x_log10(labels = lfunx), 
        Guide.xticks(ticks = log10.(gengrid([0.1,1]))),
        Guide.ylabel("V(λ)"), 
        Guide.xlabel(" λ ")
        )
    
end


function standard_plot(y, b, x, xλ, x₀)
    n = length(x) 
    d = 1:1:n 
    df1 = DataFrame(x = d, y = y, Color = ["y" for i = 1:n]) 
    df2 = DataFrame(x = d, y = b, Color = ["b" for i = 1:n]) 
    df = [df1; df2] 
    p1 = graph(df) 

    df1 = DataFrame(x = d, y = x, Color = ["x" for i = 1:n]) 
    df2 = DataFrame(x = d[1:n-1], y = xλ[1:n-1], Color = ["xλ" for i = 1:n-1]) 
    df3 = DataFrame(x = d[1:n-1], y = x₀[1:n-1], Color = ["x<sub>0</sub>" for i = 1:n-1]) 
    df = [df1; df2; df3] 
    p2 = graph(df) 

    set_default_plot_size(15cm, 6cm) 
    hstack(p1, p2) 
end

function standard_plot1(y, b, x, xλ, x₀)
    n = length(x) 
    d = 1:1:n 
    df1 = DataFrame(x = d, y = y, Color = ["y" for i = 1:n]) 
    df2 = DataFrame(x = d, y = b, Color = ["b" for i = 1:n]) 
    df = [df1; df2] 
    p1 = graph(df) 

    df1 = DataFrame(x = d, y = x, Color = ["x" for i = 1:n]) 
    df2 = DataFrame(x = d[1:n-1], y = xλ[1:n-1], Color = ["xλ1" for i = 1:n-1]) 
    df3 = DataFrame(x = d[1:n-1], y = x₀[1:n-1], Color = ["xλ2" for i = 1:n-1]) 
    df = [df1; df2; df3] 
    p2 = graph(df) 

    set_default_plot_size(15cm, 6cm) 
    hstack(p1, p2) 
end