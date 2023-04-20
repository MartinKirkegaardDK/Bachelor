/*!
 * Copyright (c) 2012 - 2021, Anaconda, Inc., and Bokeh Contributors
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 * 
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * 
 * Neither the name of Anaconda nor the names of any contributors
 * may be used to endorse or promote products derived from this software
 * without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
 * THE POSSIBILITY OF SUCH DAMAGE.
 */
(function(root, factory) {
  factory(root["Bokeh"], "2.4.2");
})(this, function(Bokeh, version) {
  let define;
  return (function(modules, entry, aliases, externals) {
    const bokeh = typeof Bokeh !== "undefined" && (version != null ? Bokeh[version] : Bokeh);
    if (bokeh != null) {
      return bokeh.register_plugin(modules, entry, aliases);
    } else {
      throw new Error("Cannot find Bokeh " + version + ". You have to load it prior to loading plugins.");
    }
  })
({
427: /* api/main.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    const tslib_1 = require(1) /* tslib */;
    (0, tslib_1.__exportStar)(require(428) /* ./index */, exports);
},
428: /* api/index.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    const tslib_1 = require(1) /* tslib */;
    const LinAlg = (0, tslib_1.__importStar)(require(429) /* ./linalg */);
    exports.LinAlg = LinAlg;
    const Charts = (0, tslib_1.__importStar)(require(430) /* ./charts */);
    exports.Charts = Charts;
    const Plotting = (0, tslib_1.__importStar)(require(433) /* ./plotting */);
    exports.Plotting = Plotting;
    const Palettes = (0, tslib_1.__importStar)(require(431) /* ./palettes */);
    exports.Palettes = Palettes;
    var document_1 = require(5) /* ../document */;
    __esExport("Document", document_1.Document);
    var templating_1 = require(152) /* ../core/util/templating */;
    __esExport("sprintf", templating_1.sprintf);
    (0, tslib_1.__exportStar)(require(432) /* ./models */, exports);
},
429: /* api/linalg.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    const tslib_1 = require(1) /* tslib */;
    var object_1 = require(13) /* ../core/util/object */;
    __esExport("keys", object_1.keys);
    __esExport("values", object_1.values);
    __esExport("entries", object_1.entries);
    __esExport("size", object_1.size);
    __esExport("extend", object_1.extend);
    (0, tslib_1.__exportStar)(require(9) /* ../core/util/array */, exports);
    (0, tslib_1.__exportStar)(require(34) /* ../core/util/string */, exports);
    (0, tslib_1.__exportStar)(require(237) /* ../core/util/random */, exports);
    (0, tslib_1.__exportStar)(require(8) /* ../core/util/types */, exports);
    (0, tslib_1.__exportStar)(require(26) /* ../core/util/eq */, exports);
},
430: /* api/charts.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    const tslib_1 = require(1) /* tslib */;
    const palettes = (0, tslib_1.__importStar)(require(431) /* ./palettes */);
    const color_1 = require(22) /* ../core/util/color */;
    const array_1 = require(9) /* ../core/util/array */;
    const types_1 = require(8) /* ../core/util/types */;
    const templating_1 = require(152) /* ../core/util/templating */;
    const models_1 = require(432) /* ./models */;
    function resolve_palette(palette = "Spectral11") {
        return (0, types_1.isArray)(palette) ? palette : palettes[palette];
    }
    function pie(data, opts = {}) {
        var _a;
        const labels = [];
        const values = [];
        for (let i = 0; i < Math.min(data.labels.length, data.values.length); i++) {
            if (data.values[i] > 0) {
                labels.push(data.labels[i]);
                values.push(data.values[i]);
            }
        }
        const start_angle = opts.start_angle != null ? opts.start_angle : 0;
        const end_angle = opts.end_angle != null ? opts.end_angle : (start_angle + 2 * Math.PI);
        const angle_span = Math.abs(end_angle - start_angle);
        const to_radians = (x) => angle_span * x;
        const total_value = (0, array_1.sum)(values);
        const normalized_values = values.map((v) => v / total_value);
        const cumulative_values = (0, array_1.cumsum)(normalized_values);
        const end_angles = cumulative_values.map((v) => start_angle + to_radians(v));
        const start_angles = [start_angle].concat(end_angles.slice(0, -1));
        const half_angles = (0, array_1.zip)(start_angles, end_angles).map(([start, end]) => (start + end) / 2);
        let cx;
        let cy;
        if (opts.center == null) {
            cx = 0;
            cy = 0;
        }
        else if ((0, types_1.isArray)(opts.center)) {
            cx = opts.center[0];
            cy = opts.center[1];
        }
        else {
            cx = opts.center.x;
            cy = opts.center.y;
        }
        const inner_radius = opts.inner_radius != null ? opts.inner_radius : 0;
        const outer_radius = opts.outer_radius != null ? opts.outer_radius : 1;
        const palette = resolve_palette(opts.palette);
        const colors = [];
        for (let i = 0; i < normalized_values.length; i++)
            colors.push(palette[i % palette.length]);
        const text_colors = colors.map((c) => (0, color_1.is_dark)((0, color_1.color2rgba)(c)) ? "white" : "black");
        function to_cartesian(r, alpha) {
            return [r * Math.cos(alpha), r * Math.sin(alpha)];
        }
        const half_radius = (inner_radius + outer_radius) / 2;
        let [text_cx, text_cy] = (0, array_1.unzip)(half_angles.map((half_angle) => to_cartesian(half_radius, half_angle)));
        text_cx = text_cx.map((x) => x + cx);
        text_cy = text_cy.map((y) => y + cy);
        const text_angles = half_angles.map((a) => {
            if (a >= Math.PI / 2 && a <= 3 * Math.PI / 2)
                return a + Math.PI;
            else
                return a;
        });
        const source = new models_1.ColumnDataSource({
            data: {
                labels,
                values,
                percentages: normalized_values.map((v) => (0, templating_1.sprintf)("%.2f%%", v * 100)),
                start_angles,
                end_angles,
                text_angles,
                colors,
                text_colors,
                text_cx,
                text_cy,
            },
        });
        const g1 = new models_1.AnnularWedge({
            x: cx, y: cy,
            inner_radius, outer_radius,
            start_angle: { field: "start_angles" }, end_angle: { field: "end_angles" },
            line_color: null, line_width: 1, fill_color: { field: "colors" },
        });
        const h1 = new models_1.AnnularWedge({
            x: cx, y: cy,
            inner_radius, outer_radius,
            start_angle: { field: "start_angles" }, end_angle: { field: "end_angles" },
            line_color: null, line_width: 1, fill_color: { field: "colors" }, fill_alpha: 0.8,
        });
        const r1 = new models_1.GlyphRenderer({
            data_source: source,
            glyph: g1,
            hover_glyph: h1,
        });
        const g2 = new models_1.Text({
            x: { field: "text_cx" }, y: { field: "text_cy" },
            text: { field: (_a = opts.slice_labels) !== null && _a !== void 0 ? _a : "labels" },
            angle: { field: "text_angles" },
            text_align: "center", text_baseline: "middle",
            text_color: { field: "text_colors" }, text_font_size: "12px",
        });
        const r2 = new models_1.GlyphRenderer({
            data_source: source,
            glyph: g2,
        });
        const xdr = new models_1.DataRange1d({ renderers: [r1], range_padding: 0.2 });
        const ydr = new models_1.DataRange1d({ renderers: [r1], range_padding: 0.2 });
        const plot = new models_1.Plot({ x_range: xdr, y_range: ydr });
        plot.add_renderers(r1, r2);
        const tooltip = "<div>@labels</div><div><b>@values</b> (@percentages)</div>";
        const hover = new models_1.HoverTool({ renderers: [r1], tooltips: tooltip });
        plot.add_tools(hover);
        return plot;
    }
    exports.pie = pie;
    function bar(data, opts = {}) {
        const column_names = data[0];
        const row_data = data.slice(1);
        const col_data = (0, array_1.transpose)(row_data);
        const labels = col_data[0].map((v) => v.toString());
        const columns = col_data.slice(1);
        let yaxis = new models_1.CategoricalAxis();
        let ydr = new models_1.FactorRange({ factors: labels });
        let yscale = new models_1.CategoricalScale();
        let xformatter;
        if (opts.axis_number_format != null)
            xformatter = new models_1.NumeralTickFormatter({ format: opts.axis_number_format });
        else
            xformatter = new models_1.BasicTickFormatter();
        let xaxis = new models_1.LinearAxis({ formatter: xformatter });
        let xdr = new models_1.DataRange1d({ start: 0 });
        let xscale = new models_1.LinearScale();
        const palette = resolve_palette(opts.palette);
        const stacked = opts.stacked != null ? opts.stacked : false;
        const orientation = opts.orientation != null ? opts.orientation : "horizontal";
        const renderers = [];
        if (stacked) {
            const left = [];
            const right = [];
            for (let i = 0; i < columns.length; i++) {
                const bottom = [];
                const top = [];
                for (let j = 0; j < labels.length; j++) {
                    const label = labels[j];
                    if (i == 0) {
                        left.push(0);
                        right.push(columns[i][j]);
                    }
                    else {
                        left[j] += columns[i - 1][j];
                        right[j] += columns[i][j];
                    }
                    bottom.push([label, -0.5]);
                    top.push([label, 0.5]);
                }
                const source = new models_1.ColumnDataSource({
                    data: {
                        left: (0, array_1.copy)(left),
                        right: (0, array_1.copy)(right),
                        top,
                        bottom,
                        labels,
                        values: columns[i],
                        columns: columns[i].map((_) => column_names[i + 1]),
                    },
                });
                const g1 = new models_1.Quad({
                    left: { field: "left" }, bottom: { field: "bottom" },
                    right: { field: "right" }, top: { field: "top" },
                    line_color: null, fill_color: palette[i % palette.length],
                });
                const r1 = new models_1.GlyphRenderer({ data_source: source, glyph: g1 });
                renderers.push(r1);
            }
        }
        else {
            const dy = 1 / columns.length;
            for (let i = 0; i < columns.length; i++) {
                const left = [];
                const right = [];
                const bottom = [];
                const top = [];
                for (let j = 0; j < labels.length; j++) {
                    const label = labels[j];
                    left.push(0);
                    right.push(columns[i][j]);
                    bottom.push([label, i * dy - 0.5]);
                    top.push([label, (i + 1) * dy - 0.5]);
                }
                const source = new models_1.ColumnDataSource({
                    data: {
                        left,
                        right,
                        top,
                        bottom,
                        labels,
                        values: columns[i],
                        columns: columns[i].map((_) => column_names[i + 1]),
                    },
                });
                const g1 = new models_1.Quad({
                    left: { field: "left" }, bottom: { field: "bottom" },
                    right: { field: "right" }, top: { field: "top" },
                    line_color: null, fill_color: palette[i % palette.length],
                });
                const r1 = new models_1.GlyphRenderer({ data_source: source, glyph: g1 });
                renderers.push(r1);
            }
        }
        if (orientation == "vertical") {
            [xdr, ydr] = [ydr, xdr];
            [xaxis, yaxis] = [yaxis, xaxis];
            [xscale, yscale] = [yscale, xscale];
            for (const r of renderers) {
                const data = r.data_source.data;
                [data.left, data.bottom] = [data.bottom, data.left];
                [data.right, data.top] = [data.top, data.right];
            }
        }
        const plot = new models_1.Plot({ x_range: xdr, y_range: ydr, x_scale: xscale, y_scale: yscale });
        plot.add_renderers(...renderers);
        plot.add_layout(yaxis, "left");
        plot.add_layout(xaxis, "below");
        const tooltip = "<div>@labels</div><div>@columns:&nbsp<b>@values</b></div>";
        let anchor;
        let attachment;
        if (orientation == "horizontal") {
            anchor = "center_right";
            attachment = "horizontal";
        }
        else {
            anchor = "top_center";
            attachment = "vertical";
        }
        const hover = new models_1.HoverTool({
            renderers,
            tooltips: tooltip,
            point_policy: "snap_to_data",
            anchor,
            attachment,
        });
        plot.add_tools(hover);
        return plot;
    }
    exports.bar = bar;
},
431: /* api/palettes.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    exports.Greens4 = exports.Greens3 = exports.Blues9 = exports.Blues8 = exports.Blues7 = exports.Blues6 = exports.Blues5 = exports.Blues4 = exports.Blues3 = exports.Purples9 = exports.Purples8 = exports.Purples7 = exports.Purples6 = exports.Purples5 = exports.Purples4 = exports.Purples3 = exports.YlOrBr9 = exports.YlOrBr8 = exports.YlOrBr7 = exports.YlOrBr6 = exports.YlOrBr5 = exports.YlOrBr4 = exports.YlOrBr3 = exports.YlOrRd9 = exports.YlOrRd8 = exports.YlOrRd7 = exports.YlOrRd6 = exports.YlOrRd5 = exports.YlOrRd4 = exports.YlOrRd3 = exports.OrRd9 = exports.OrRd8 = exports.OrRd7 = exports.OrRd6 = exports.OrRd5 = exports.OrRd4 = exports.OrRd3 = exports.PuRd9 = exports.PuRd8 = exports.PuRd7 = exports.PuRd6 = exports.PuRd5 = exports.PuRd4 = exports.PuRd3 = exports.RdPu9 = exports.RdPu8 = exports.RdPu7 = exports.RdPu6 = exports.RdPu5 = exports.RdPu4 = void 0;
    exports.PRGn5 = exports.PRGn4 = exports.PRGn3 = exports.BrBG11 = exports.BrBG10 = exports.BrBG9 = exports.BrBG8 = exports.BrBG7 = exports.BrBG6 = exports.BrBG5 = exports.BrBG4 = exports.BrBG3 = exports.PuOr11 = exports.PuOr10 = exports.PuOr9 = exports.PuOr8 = exports.PuOr7 = exports.PuOr6 = exports.PuOr5 = exports.PuOr4 = exports.PuOr3 = exports.Greys256 = exports.Greys11 = exports.Greys10 = exports.Greys9 = exports.Greys8 = exports.Greys7 = exports.Greys6 = exports.Greys5 = exports.Greys4 = exports.Greys3 = exports.Reds9 = exports.Reds8 = exports.Reds7 = exports.Reds6 = exports.Reds5 = exports.Reds4 = exports.Reds3 = exports.Oranges9 = exports.Oranges8 = exports.Oranges7 = exports.Oranges6 = exports.Oranges5 = exports.Oranges4 = exports.Oranges3 = exports.Greens9 = exports.Greens8 = exports.Greens7 = exports.Greens6 = exports.Greens5 = void 0;
    exports.Spectral10 = exports.Spectral9 = exports.Spectral8 = exports.Spectral7 = exports.Spectral6 = exports.Spectral5 = exports.Spectral4 = exports.Spectral3 = exports.RdYlBu11 = exports.RdYlBu10 = exports.RdYlBu9 = exports.RdYlBu8 = exports.RdYlBu7 = exports.RdYlBu6 = exports.RdYlBu5 = exports.RdYlBu4 = exports.RdYlBu3 = exports.RdGy11 = exports.RdGy10 = exports.RdGy9 = exports.RdGy8 = exports.RdGy7 = exports.RdGy6 = exports.RdGy5 = exports.RdGy4 = exports.RdGy3 = exports.RdBu11 = exports.RdBu10 = exports.RdBu9 = exports.RdBu8 = exports.RdBu7 = exports.RdBu6 = exports.RdBu5 = exports.RdBu4 = exports.RdBu3 = exports.PiYG11 = exports.PiYG10 = exports.PiYG9 = exports.PiYG8 = exports.PiYG7 = exports.PiYG6 = exports.PiYG5 = exports.PiYG4 = exports.PiYG3 = exports.PRGn11 = exports.PRGn10 = exports.PRGn9 = exports.PRGn8 = exports.PRGn7 = exports.PRGn6 = void 0;
    exports.Viridis256 = exports.Viridis11 = exports.Viridis10 = exports.Viridis9 = exports.Viridis8 = exports.Viridis7 = exports.Viridis6 = exports.Viridis5 = exports.Viridis4 = exports.Viridis3 = exports.Plasma256 = exports.Plasma11 = exports.Plasma10 = exports.Plasma9 = exports.Plasma8 = exports.Plasma7 = exports.Plasma6 = exports.Plasma5 = exports.Plasma4 = exports.Plasma3 = exports.Magma256 = exports.Magma11 = exports.Magma10 = exports.Magma9 = exports.Magma8 = exports.Magma7 = exports.Magma6 = exports.Magma5 = exports.Magma4 = exports.Magma3 = exports.Inferno256 = exports.Inferno11 = exports.Inferno10 = exports.Inferno9 = exports.Inferno8 = exports.Inferno7 = exports.Inferno6 = exports.Inferno5 = exports.Inferno4 = exports.Inferno3 = exports.RdYlGn11 = exports.RdYlGn10 = exports.RdYlGn9 = exports.RdYlGn8 = exports.RdYlGn7 = exports.RdYlGn6 = exports.RdYlGn5 = exports.RdYlGn4 = exports.RdYlGn3 = exports.Spectral11 = void 0;
    exports.Pastel2_3 = exports.Pastel1_9 = exports.Pastel1_8 = exports.Pastel1_7 = exports.Pastel1_6 = exports.Pastel1_5 = exports.Pastel1_4 = exports.Pastel1_3 = exports.Paired12 = exports.Paired11 = exports.Paired10 = exports.Paired9 = exports.Paired8 = exports.Paired7 = exports.Paired6 = exports.Paired5 = exports.Paired4 = exports.Paired3 = exports.Dark2_8 = exports.Dark2_7 = exports.Dark2_6 = exports.Dark2_5 = exports.Dark2_4 = exports.Dark2_3 = exports.Accent8 = exports.Accent7 = exports.Accent6 = exports.Accent5 = exports.Accent4 = exports.Accent3 = exports.Turbo256 = exports.Turbo11 = exports.Turbo10 = exports.Turbo9 = exports.Turbo8 = exports.Turbo7 = exports.Turbo6 = exports.Turbo5 = exports.Turbo4 = exports.Turbo3 = exports.Cividis256 = exports.Cividis11 = exports.Cividis10 = exports.Cividis9 = exports.Cividis8 = exports.Cividis7 = exports.Cividis6 = exports.Cividis5 = exports.Cividis4 = exports.Cividis3 = void 0;
    exports.Category20_16 = exports.Category20_15 = exports.Category20_14 = exports.Category20_13 = exports.Category20_12 = exports.Category20_11 = exports.Category20_10 = exports.Category20_9 = exports.Category20_8 = exports.Category20_7 = exports.Category20_6 = exports.Category20_5 = exports.Category20_4 = exports.Category20_3 = exports.Category10_10 = exports.Category10_9 = exports.Category10_8 = exports.Category10_7 = exports.Category10_6 = exports.Category10_5 = exports.Category10_4 = exports.Category10_3 = exports.Set3_12 = exports.Set3_11 = exports.Set3_10 = exports.Set3_9 = exports.Set3_8 = exports.Set3_7 = exports.Set3_6 = exports.Set3_5 = exports.Set3_4 = exports.Set3_3 = exports.Set2_8 = exports.Set2_7 = exports.Set2_6 = exports.Set2_5 = exports.Set2_4 = exports.Set2_3 = exports.Set1_9 = exports.Set1_8 = exports.Set1_7 = exports.Set1_6 = exports.Set1_5 = exports.Set1_4 = exports.Set1_3 = exports.Pastel2_8 = exports.Pastel2_7 = exports.Pastel2_6 = exports.Pastel2_5 = exports.Pastel2_4 = void 0;
    exports.BuGn = exports.GnBu = exports.YlGnBu = exports.YlGn = exports.Colorblind8 = exports.Colorblind7 = exports.Colorblind6 = exports.Colorblind5 = exports.Colorblind4 = exports.Colorblind3 = exports.Category20c_20 = exports.Category20c_19 = exports.Category20c_18 = exports.Category20c_17 = exports.Category20c_16 = exports.Category20c_15 = exports.Category20c_14 = exports.Category20c_13 = exports.Category20c_12 = exports.Category20c_11 = exports.Category20c_10 = exports.Category20c_9 = exports.Category20c_8 = exports.Category20c_7 = exports.Category20c_6 = exports.Category20c_5 = exports.Category20c_4 = exports.Category20c_3 = exports.Category20b_20 = exports.Category20b_19 = exports.Category20b_18 = exports.Category20b_17 = exports.Category20b_16 = exports.Category20b_15 = exports.Category20b_14 = exports.Category20b_13 = exports.Category20b_12 = exports.Category20b_11 = exports.Category20b_10 = exports.Category20b_9 = exports.Category20b_8 = exports.Category20b_7 = exports.Category20b_6 = exports.Category20b_5 = exports.Category20b_4 = exports.Category20b_3 = exports.Category20_20 = exports.Category20_19 = exports.Category20_18 = exports.Category20_17 = void 0;
    exports.grey = exports.turbo = exports.cividis = exports.viridis = exports.plasma = exports.inferno = exports.magma = exports.linear_palette = exports.Colorblind = exports.Category20c = exports.Category20b = exports.Category20 = exports.Category10 = exports.Set3 = exports.Set2 = exports.Set1 = exports.Pastel2 = exports.Pastel1 = exports.Paired = exports.Dark2 = exports.Accent = exports.Viridis = exports.Plasma = exports.Magma = exports.Inferno = exports.RdYlGn = exports.Spectral = exports.RdYlBu = exports.RdGy = exports.RdBu = exports.PiYG = exports.PRGn = exports.BrBG = exports.PuOr = exports.Greys = exports.Reds = exports.Oranges = exports.Greens = exports.Blues = exports.Purples = exports.YlOrBr = exports.YlOrRd = exports.OrRd = exports.PuRd = exports.RdPu = exports.BuPu = exports.PuBu = exports.PuBuGn = void 0;
    const array_1 = require(9) /* ../core/util/array */;
    exports.YlGn3 = [0x31a354ff, 0xaddd8eff, 0xf7fcb9ff];
    exports.YlGn4 = [0x238443ff, 0x78c679ff, 0xc2e699ff, 0xffffccff];
    exports.YlGn5 = [0x006837ff, 0x31a354ff, 0x78c679ff, 0xc2e699ff, 0xffffccff];
    exports.YlGn6 = [0x006837ff, 0x31a354ff, 0x78c679ff, 0xaddd8eff, 0xd9f0a3ff, 0xffffccff];
    exports.YlGn7 = [0x005a32ff, 0x238443ff, 0x41ab5dff, 0x78c679ff, 0xaddd8eff, 0xd9f0a3ff, 0xffffccff];
    exports.YlGn8 = [0x005a32ff, 0x238443ff, 0x41ab5dff, 0x78c679ff, 0xaddd8eff, 0xd9f0a3ff, 0xf7fcb9ff, 0xffffe5ff];
    exports.YlGn9 = [0x004529ff, 0x006837ff, 0x238443ff, 0x41ab5dff, 0x78c679ff, 0xaddd8eff, 0xd9f0a3ff, 0xf7fcb9ff, 0xffffe5ff];
    exports.YlGnBu3 = [0x2c7fb8ff, 0x7fcdbbff, 0xedf8b1ff];
    exports.YlGnBu4 = [0x225ea8ff, 0x41b6c4ff, 0xa1dab4ff, 0xffffccff];
    exports.YlGnBu5 = [0x253494ff, 0x2c7fb8ff, 0x41b6c4ff, 0xa1dab4ff, 0xffffccff];
    exports.YlGnBu6 = [0x253494ff, 0x2c7fb8ff, 0x41b6c4ff, 0x7fcdbbff, 0xc7e9b4ff, 0xffffccff];
    exports.YlGnBu7 = [0x0c2c84ff, 0x225ea8ff, 0x1d91c0ff, 0x41b6c4ff, 0x7fcdbbff, 0xc7e9b4ff, 0xffffccff];
    exports.YlGnBu8 = [0x0c2c84ff, 0x225ea8ff, 0x1d91c0ff, 0x41b6c4ff, 0x7fcdbbff, 0xc7e9b4ff, 0xedf8b1ff, 0xffffd9ff];
    exports.YlGnBu9 = [0x081d58ff, 0x253494ff, 0x225ea8ff, 0x1d91c0ff, 0x41b6c4ff, 0x7fcdbbff, 0xc7e9b4ff, 0xedf8b1ff, 0xffffd9ff];
    exports.GnBu3 = [0x43a2caff, 0xa8ddb5ff, 0xe0f3dbff];
    exports.GnBu4 = [0x2b8cbeff, 0x7bccc4ff, 0xbae4bcff, 0xf0f9e8ff];
    exports.GnBu5 = [0x0868acff, 0x43a2caff, 0x7bccc4ff, 0xbae4bcff, 0xf0f9e8ff];
    exports.GnBu6 = [0x0868acff, 0x43a2caff, 0x7bccc4ff, 0xa8ddb5ff, 0xccebc5ff, 0xf0f9e8ff];
    exports.GnBu7 = [0x08589eff, 0x2b8cbeff, 0x4eb3d3ff, 0x7bccc4ff, 0xa8ddb5ff, 0xccebc5ff, 0xf0f9e8ff];
    exports.GnBu8 = [0x08589eff, 0x2b8cbeff, 0x4eb3d3ff, 0x7bccc4ff, 0xa8ddb5ff, 0xccebc5ff, 0xe0f3dbff, 0xf7fcf0ff];
    exports.GnBu9 = [0x084081ff, 0x0868acff, 0x2b8cbeff, 0x4eb3d3ff, 0x7bccc4ff, 0xa8ddb5ff, 0xccebc5ff, 0xe0f3dbff, 0xf7fcf0ff];
    exports.BuGn3 = [0x2ca25fff, 0x99d8c9ff, 0xe5f5f9ff];
    exports.BuGn4 = [0x238b45ff, 0x66c2a4ff, 0xb2e2e2ff, 0xedf8fbff];
    exports.BuGn5 = [0x006d2cff, 0x2ca25fff, 0x66c2a4ff, 0xb2e2e2ff, 0xedf8fbff];
    exports.BuGn6 = [0x006d2cff, 0x2ca25fff, 0x66c2a4ff, 0x99d8c9ff, 0xccece6ff, 0xedf8fbff];
    exports.BuGn7 = [0x005824ff, 0x238b45ff, 0x41ae76ff, 0x66c2a4ff, 0x99d8c9ff, 0xccece6ff, 0xedf8fbff];
    exports.BuGn8 = [0x005824ff, 0x238b45ff, 0x41ae76ff, 0x66c2a4ff, 0x99d8c9ff, 0xccece6ff, 0xe5f5f9ff, 0xf7fcfdff];
    exports.BuGn9 = [0x00441bff, 0x006d2cff, 0x238b45ff, 0x41ae76ff, 0x66c2a4ff, 0x99d8c9ff, 0xccece6ff, 0xe5f5f9ff, 0xf7fcfdff];
    exports.PuBuGn3 = [0x1c9099ff, 0xa6bddbff, 0xece2f0ff];
    exports.PuBuGn4 = [0x02818aff, 0x67a9cfff, 0xbdc9e1ff, 0xf6eff7ff];
    exports.PuBuGn5 = [0x016c59ff, 0x1c9099ff, 0x67a9cfff, 0xbdc9e1ff, 0xf6eff7ff];
    exports.PuBuGn6 = [0x016c59ff, 0x1c9099ff, 0x67a9cfff, 0xa6bddbff, 0xd0d1e6ff, 0xf6eff7ff];
    exports.PuBuGn7 = [0x016450ff, 0x02818aff, 0x3690c0ff, 0x67a9cfff, 0xa6bddbff, 0xd0d1e6ff, 0xf6eff7ff];
    exports.PuBuGn8 = [0x016450ff, 0x02818aff, 0x3690c0ff, 0x67a9cfff, 0xa6bddbff, 0xd0d1e6ff, 0xece2f0ff, 0xfff7fbff];
    exports.PuBuGn9 = [0x014636ff, 0x016c59ff, 0x02818aff, 0x3690c0ff, 0x67a9cfff, 0xa6bddbff, 0xd0d1e6ff, 0xece2f0ff, 0xfff7fbff];
    exports.PuBu3 = [0x2b8cbeff, 0xa6bddbff, 0xece7f2ff];
    exports.PuBu4 = [0x0570b0ff, 0x74a9cfff, 0xbdc9e1ff, 0xf1eef6ff];
    exports.PuBu5 = [0x045a8dff, 0x2b8cbeff, 0x74a9cfff, 0xbdc9e1ff, 0xf1eef6ff];
    exports.PuBu6 = [0x045a8dff, 0x2b8cbeff, 0x74a9cfff, 0xa6bddbff, 0xd0d1e6ff, 0xf1eef6ff];
    exports.PuBu7 = [0x034e7bff, 0x0570b0ff, 0x3690c0ff, 0x74a9cfff, 0xa6bddbff, 0xd0d1e6ff, 0xf1eef6ff];
    exports.PuBu8 = [0x034e7bff, 0x0570b0ff, 0x3690c0ff, 0x74a9cfff, 0xa6bddbff, 0xd0d1e6ff, 0xece7f2ff, 0xfff7fbff];
    exports.PuBu9 = [0x023858ff, 0x045a8dff, 0x0570b0ff, 0x3690c0ff, 0x74a9cfff, 0xa6bddbff, 0xd0d1e6ff, 0xece7f2ff, 0xfff7fbff];
    exports.BuPu3 = [0x8856a7ff, 0x9ebcdaff, 0xe0ecf4ff];
    exports.BuPu4 = [0x88419dff, 0x8c96c6ff, 0xb3cde3ff, 0xedf8fbff];
    exports.BuPu5 = [0x810f7cff, 0x8856a7ff, 0x8c96c6ff, 0xb3cde3ff, 0xedf8fbff];
    exports.BuPu6 = [0x810f7cff, 0x8856a7ff, 0x8c96c6ff, 0x9ebcdaff, 0xbfd3e6ff, 0xedf8fbff];
    exports.BuPu7 = [0x6e016bff, 0x88419dff, 0x8c6bb1ff, 0x8c96c6ff, 0x9ebcdaff, 0xbfd3e6ff, 0xedf8fbff];
    exports.BuPu8 = [0x6e016bff, 0x88419dff, 0x8c6bb1ff, 0x8c96c6ff, 0x9ebcdaff, 0xbfd3e6ff, 0xe0ecf4ff, 0xf7fcfdff];
    exports.BuPu9 = [0x4d004bff, 0x810f7cff, 0x88419dff, 0x8c6bb1ff, 0x8c96c6ff, 0x9ebcdaff, 0xbfd3e6ff, 0xe0ecf4ff, 0xf7fcfdff];
    exports.RdPu3 = [0xc51b8aff, 0xfa9fb5ff, 0xfde0ddff];
    exports.RdPu4 = [0xae017eff, 0xf768a1ff, 0xfbb4b9ff, 0xfeebe2ff];
    exports.RdPu5 = [0x7a0177ff, 0xc51b8aff, 0xf768a1ff, 0xfbb4b9ff, 0xfeebe2ff];
    exports.RdPu6 = [0x7a0177ff, 0xc51b8aff, 0xf768a1ff, 0xfa9fb5ff, 0xfcc5c0ff, 0xfeebe2ff];
    exports.RdPu7 = [0x7a0177ff, 0xae017eff, 0xdd3497ff, 0xf768a1ff, 0xfa9fb5ff, 0xfcc5c0ff, 0xfeebe2ff];
    exports.RdPu8 = [0x7a0177ff, 0xae017eff, 0xdd3497ff, 0xf768a1ff, 0xfa9fb5ff, 0xfcc5c0ff, 0xfde0ddff, 0xfff7f3ff];
    exports.RdPu9 = [0x49006aff, 0x7a0177ff, 0xae017eff, 0xdd3497ff, 0xf768a1ff, 0xfa9fb5ff, 0xfcc5c0ff, 0xfde0ddff, 0xfff7f3ff];
    exports.PuRd3 = [0xdd1c77ff, 0xc994c7ff, 0xe7e1efff];
    exports.PuRd4 = [0xce1256ff, 0xdf65b0ff, 0xd7b5d8ff, 0xf1eef6ff];
    exports.PuRd5 = [0x980043ff, 0xdd1c77ff, 0xdf65b0ff, 0xd7b5d8ff, 0xf1eef6ff];
    exports.PuRd6 = [0x980043ff, 0xdd1c77ff, 0xdf65b0ff, 0xc994c7ff, 0xd4b9daff, 0xf1eef6ff];
    exports.PuRd7 = [0x91003fff, 0xce1256ff, 0xe7298aff, 0xdf65b0ff, 0xc994c7ff, 0xd4b9daff, 0xf1eef6ff];
    exports.PuRd8 = [0x91003fff, 0xce1256ff, 0xe7298aff, 0xdf65b0ff, 0xc994c7ff, 0xd4b9daff, 0xe7e1efff, 0xf7f4f9ff];
    exports.PuRd9 = [0x67001fff, 0x980043ff, 0xce1256ff, 0xe7298aff, 0xdf65b0ff, 0xc994c7ff, 0xd4b9daff, 0xe7e1efff, 0xf7f4f9ff];
    exports.OrRd3 = [0xe34a33ff, 0xfdbb84ff, 0xfee8c8ff];
    exports.OrRd4 = [0xd7301fff, 0xfc8d59ff, 0xfdcc8aff, 0xfef0d9ff];
    exports.OrRd5 = [0xb30000ff, 0xe34a33ff, 0xfc8d59ff, 0xfdcc8aff, 0xfef0d9ff];
    exports.OrRd6 = [0xb30000ff, 0xe34a33ff, 0xfc8d59ff, 0xfdbb84ff, 0xfdd49eff, 0xfef0d9ff];
    exports.OrRd7 = [0x990000ff, 0xd7301fff, 0xef6548ff, 0xfc8d59ff, 0xfdbb84ff, 0xfdd49eff, 0xfef0d9ff];
    exports.OrRd8 = [0x990000ff, 0xd7301fff, 0xef6548ff, 0xfc8d59ff, 0xfdbb84ff, 0xfdd49eff, 0xfee8c8ff, 0xfff7ecff];
    exports.OrRd9 = [0x7f0000ff, 0xb30000ff, 0xd7301fff, 0xef6548ff, 0xfc8d59ff, 0xfdbb84ff, 0xfdd49eff, 0xfee8c8ff, 0xfff7ecff];
    exports.YlOrRd3 = [0xf03b20ff, 0xfeb24cff, 0xffeda0ff];
    exports.YlOrRd4 = [0xe31a1cff, 0xfd8d3cff, 0xfecc5cff, 0xffffb2ff];
    exports.YlOrRd5 = [0xbd0026ff, 0xf03b20ff, 0xfd8d3cff, 0xfecc5cff, 0xffffb2ff];
    exports.YlOrRd6 = [0xbd0026ff, 0xf03b20ff, 0xfd8d3cff, 0xfeb24cff, 0xfed976ff, 0xffffb2ff];
    exports.YlOrRd7 = [0xb10026ff, 0xe31a1cff, 0xfc4e2aff, 0xfd8d3cff, 0xfeb24cff, 0xfed976ff, 0xffffb2ff];
    exports.YlOrRd8 = [0xb10026ff, 0xe31a1cff, 0xfc4e2aff, 0xfd8d3cff, 0xfeb24cff, 0xfed976ff, 0xffeda0ff, 0xffffccff];
    exports.YlOrRd9 = [0x800026ff, 0xbd0026ff, 0xe31a1cff, 0xfc4e2aff, 0xfd8d3cff, 0xfeb24cff, 0xfed976ff, 0xffeda0ff, 0xffffccff];
    exports.YlOrBr3 = [0xd95f0eff, 0xfec44fff, 0xfff7bcff];
    exports.YlOrBr4 = [0xcc4c02ff, 0xfe9929ff, 0xfed98eff, 0xffffd4ff];
    exports.YlOrBr5 = [0x993404ff, 0xd95f0eff, 0xfe9929ff, 0xfed98eff, 0xffffd4ff];
    exports.YlOrBr6 = [0x993404ff, 0xd95f0eff, 0xfe9929ff, 0xfec44fff, 0xfee391ff, 0xffffd4ff];
    exports.YlOrBr7 = [0x8c2d04ff, 0xcc4c02ff, 0xec7014ff, 0xfe9929ff, 0xfec44fff, 0xfee391ff, 0xffffd4ff];
    exports.YlOrBr8 = [0x8c2d04ff, 0xcc4c02ff, 0xec7014ff, 0xfe9929ff, 0xfec44fff, 0xfee391ff, 0xfff7bcff, 0xffffe5ff];
    exports.YlOrBr9 = [0x662506ff, 0x993404ff, 0xcc4c02ff, 0xec7014ff, 0xfe9929ff, 0xfec44fff, 0xfee391ff, 0xfff7bcff, 0xffffe5ff];
    exports.Purples3 = [0x756bb1ff, 0xbcbddcff, 0xefedf5ff];
    exports.Purples4 = [0x6a51a3ff, 0x9e9ac8ff, 0xcbc9e2ff, 0xf2f0f7ff];
    exports.Purples5 = [0x54278fff, 0x756bb1ff, 0x9e9ac8ff, 0xcbc9e2ff, 0xf2f0f7ff];
    exports.Purples6 = [0x54278fff, 0x756bb1ff, 0x9e9ac8ff, 0xbcbddcff, 0xdadaebff, 0xf2f0f7ff];
    exports.Purples7 = [0x4a1486ff, 0x6a51a3ff, 0x807dbaff, 0x9e9ac8ff, 0xbcbddcff, 0xdadaebff, 0xf2f0f7ff];
    exports.Purples8 = [0x4a1486ff, 0x6a51a3ff, 0x807dbaff, 0x9e9ac8ff, 0xbcbddcff, 0xdadaebff, 0xefedf5ff, 0xfcfbfdff];
    exports.Purples9 = [0x3f007dff, 0x54278fff, 0x6a51a3ff, 0x807dbaff, 0x9e9ac8ff, 0xbcbddcff, 0xdadaebff, 0xefedf5ff, 0xfcfbfdff];
    exports.Blues3 = [0x3182bdff, 0x9ecae1ff, 0xdeebf7ff];
    exports.Blues4 = [0x2171b5ff, 0x6baed6ff, 0xbdd7e7ff, 0xeff3ffff];
    exports.Blues5 = [0x08519cff, 0x3182bdff, 0x6baed6ff, 0xbdd7e7ff, 0xeff3ffff];
    exports.Blues6 = [0x08519cff, 0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xeff3ffff];
    exports.Blues7 = [0x084594ff, 0x2171b5ff, 0x4292c6ff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xeff3ffff];
    exports.Blues8 = [0x084594ff, 0x2171b5ff, 0x4292c6ff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xdeebf7ff, 0xf7fbffff];
    exports.Blues9 = [0x08306bff, 0x08519cff, 0x2171b5ff, 0x4292c6ff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xdeebf7ff, 0xf7fbffff];
    exports.Greens3 = [0x31a354ff, 0xa1d99bff, 0xe5f5e0ff];
    exports.Greens4 = [0x238b45ff, 0x74c476ff, 0xbae4b3ff, 0xedf8e9ff];
    exports.Greens5 = [0x006d2cff, 0x31a354ff, 0x74c476ff, 0xbae4b3ff, 0xedf8e9ff];
    exports.Greens6 = [0x006d2cff, 0x31a354ff, 0x74c476ff, 0xa1d99bff, 0xc7e9c0ff, 0xedf8e9ff];
    exports.Greens7 = [0x005a32ff, 0x238b45ff, 0x41ab5dff, 0x74c476ff, 0xa1d99bff, 0xc7e9c0ff, 0xedf8e9ff];
    exports.Greens8 = [0x005a32ff, 0x238b45ff, 0x41ab5dff, 0x74c476ff, 0xa1d99bff, 0xc7e9c0ff, 0xe5f5e0ff, 0xf7fcf5ff];
    exports.Greens9 = [0x00441bff, 0x006d2cff, 0x238b45ff, 0x41ab5dff, 0x74c476ff, 0xa1d99bff, 0xc7e9c0ff, 0xe5f5e0ff, 0xf7fcf5ff];
    exports.Oranges3 = [0xe6550dff, 0xfdae6bff, 0xfee6ceff];
    exports.Oranges4 = [0xd94701ff, 0xfd8d3cff, 0xfdbe85ff, 0xfeeddeff];
    exports.Oranges5 = [0xa63603ff, 0xe6550dff, 0xfd8d3cff, 0xfdbe85ff, 0xfeeddeff];
    exports.Oranges6 = [0xa63603ff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0xfeeddeff];
    exports.Oranges7 = [0x8c2d04ff, 0xd94801ff, 0xf16913ff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0xfeeddeff];
    exports.Oranges8 = [0x8c2d04ff, 0xd94801ff, 0xf16913ff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0xfee6ceff, 0xfff5ebff];
    exports.Oranges9 = [0x7f2704ff, 0xa63603ff, 0xd94801ff, 0xf16913ff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0xfee6ceff, 0xfff5ebff];
    exports.Reds3 = [0xde2d26ff, 0xfc9272ff, 0xfee0d2ff];
    exports.Reds4 = [0xcb181dff, 0xfb6a4aff, 0xfcae91ff, 0xfee5d9ff];
    exports.Reds5 = [0xa50f15ff, 0xde2d26ff, 0xfb6a4aff, 0xfcae91ff, 0xfee5d9ff];
    exports.Reds6 = [0xa50f15ff, 0xde2d26ff, 0xfb6a4aff, 0xfc9272ff, 0xfcbba1ff, 0xfee5d9ff];
    exports.Reds7 = [0x99000dff, 0xcb181dff, 0xef3b2cff, 0xfb6a4aff, 0xfc9272ff, 0xfcbba1ff, 0xfee5d9ff];
    exports.Reds8 = [0x99000dff, 0xcb181dff, 0xef3b2cff, 0xfb6a4aff, 0xfc9272ff, 0xfcbba1ff, 0xfee0d2ff, 0xfff5f0ff];
    exports.Reds9 = [0x67000dff, 0xa50f15ff, 0xcb181dff, 0xef3b2cff, 0xfb6a4aff, 0xfc9272ff, 0xfcbba1ff, 0xfee0d2ff, 0xfff5f0ff];
    exports.Greys3 = [0x636363ff, 0xbdbdbdff, 0xf0f0f0ff];
    exports.Greys4 = [0x525252ff, 0x969696ff, 0xccccccff, 0xf7f7f7ff];
    exports.Greys5 = [0x252525ff, 0x636363ff, 0x969696ff, 0xccccccff, 0xf7f7f7ff];
    exports.Greys6 = [0x252525ff, 0x636363ff, 0x969696ff, 0xbdbdbdff, 0xd9d9d9ff, 0xf7f7f7ff];
    exports.Greys7 = [0x252525ff, 0x525252ff, 0x737373ff, 0x969696ff, 0xbdbdbdff, 0xd9d9d9ff, 0xf7f7f7ff];
    exports.Greys8 = [0x252525ff, 0x525252ff, 0x737373ff, 0x969696ff, 0xbdbdbdff, 0xd9d9d9ff, 0xf0f0f0ff, 0xffffffff];
    exports.Greys9 = [0x000000ff, 0x252525ff, 0x525252ff, 0x737373ff, 0x969696ff, 0xbdbdbdff, 0xd9d9d9ff, 0xf0f0f0ff, 0xffffffff];
    exports.Greys10 = [0x000000ff, 0x1c1c1cff, 0x383838ff, 0x555555ff, 0x717171ff, 0x8d8d8dff, 0xaaaaaaff, 0xc6c6c6ff, 0xe2e2e2ff, 0xffffffff];
    exports.Greys11 = [0x000000ff, 0x191919ff, 0x333333ff, 0x4c4c4cff, 0x666666ff, 0x7f7f7fff, 0x999999ff, 0xb2b2b2ff, 0xccccccff, 0xe5e5e5ff, 0xffffffff];
    exports.Greys256 = [0x000000ff, 0x010101ff, 0x020202ff, 0x030303ff, 0x040404ff, 0x050505ff, 0x060606ff, 0x070707ff, 0x080808ff, 0x090909ff, 0x0a0a0aff, 0x0b0b0bff,
        0x0c0c0cff, 0x0d0d0dff, 0x0e0e0eff, 0x0f0f0fff, 0x101010ff, 0x111111ff, 0x121212ff, 0x131313ff, 0x141414ff, 0x151515ff, 0x161616ff, 0x171717ff,
        0x181818ff, 0x191919ff, 0x1a1a1aff, 0x1b1b1bff, 0x1c1c1cff, 0x1d1d1dff, 0x1e1e1eff, 0x1f1f1fff, 0x202020ff, 0x212121ff, 0x222222ff, 0x232323ff,
        0x242424ff, 0x252525ff, 0x262626ff, 0x272727ff, 0x282828ff, 0x292929ff, 0x2a2a2aff, 0x2b2b2bff, 0x2c2c2cff, 0x2d2d2dff, 0x2e2e2eff, 0x2f2f2fff,
        0x303030ff, 0x313131ff, 0x323232ff, 0x333333ff, 0x343434ff, 0x353535ff, 0x363636ff, 0x373737ff, 0x383838ff, 0x393939ff, 0x3a3a3aff, 0x3b3b3bff,
        0x3c3c3cff, 0x3d3d3dff, 0x3e3e3eff, 0x3f3f3fff, 0x404040ff, 0x414141ff, 0x424242ff, 0x434343ff, 0x444444ff, 0x454545ff, 0x464646ff, 0x474747ff,
        0x484848ff, 0x494949ff, 0x4a4a4aff, 0x4b4b4bff, 0x4c4c4cff, 0x4d4d4dff, 0x4e4e4eff, 0x4f4f4fff, 0x505050ff, 0x515151ff, 0x525252ff, 0x535353ff,
        0x545454ff, 0x555555ff, 0x565656ff, 0x575757ff, 0x585858ff, 0x595959ff, 0x5a5a5aff, 0x5b5b5bff, 0x5c5c5cff, 0x5d5d5dff, 0x5e5e5eff, 0x5f5f5fff,
        0x606060ff, 0x616161ff, 0x626262ff, 0x636363ff, 0x646464ff, 0x656565ff, 0x666666ff, 0x676767ff, 0x686868ff, 0x696969ff, 0x6a6a6aff, 0x6b6b6bff,
        0x6c6c6cff, 0x6d6d6dff, 0x6e6e6eff, 0x6f6f6fff, 0x707070ff, 0x717171ff, 0x727272ff, 0x737373ff, 0x747474ff, 0x757575ff, 0x767676ff, 0x777777ff,
        0x787878ff, 0x797979ff, 0x7a7a7aff, 0x7b7b7bff, 0x7c7c7cff, 0x7d7d7dff, 0x7e7e7eff, 0x7f7f7fff, 0x808080ff, 0x818181ff, 0x828282ff, 0x838383ff,
        0x848484ff, 0x858585ff, 0x868686ff, 0x878787ff, 0x888888ff, 0x898989ff, 0x8a8a8aff, 0x8b8b8bff, 0x8c8c8cff, 0x8d8d8dff, 0x8e8e8eff, 0x8f8f8fff,
        0x909090ff, 0x919191ff, 0x929292ff, 0x939393ff, 0x949494ff, 0x959595ff, 0x969696ff, 0x979797ff, 0x989898ff, 0x999999ff, 0x9a9a9aff, 0x9b9b9bff,
        0x9c9c9cff, 0x9d9d9dff, 0x9e9e9eff, 0x9f9f9fff, 0xa0a0a0ff, 0xa1a1a1ff, 0xa2a2a2ff, 0xa3a3a3ff, 0xa4a4a4ff, 0xa5a5a5ff, 0xa6a6a6ff, 0xa7a7a7ff,
        0xa8a8a8ff, 0xa9a9a9ff, 0xaaaaaaff, 0xabababff, 0xacacacff, 0xadadadff, 0xaeaeaeff, 0xafafafff, 0xb0b0b0ff, 0xb1b1b1ff, 0xb2b2b2ff, 0xb3b3b3ff,
        0xb4b4b4ff, 0xb5b5b5ff, 0xb6b6b6ff, 0xb7b7b7ff, 0xb8b8b8ff, 0xb9b9b9ff, 0xbababaff, 0xbbbbbbff, 0xbcbcbcff, 0xbdbdbdff, 0xbebebeff, 0xbfbfbfff,
        0xc0c0c0ff, 0xc1c1c1ff, 0xc2c2c2ff, 0xc3c3c3ff, 0xc4c4c4ff, 0xc5c5c5ff, 0xc6c6c6ff, 0xc7c7c7ff, 0xc8c8c8ff, 0xc9c9c9ff, 0xcacacaff, 0xcbcbcbff,
        0xccccccff, 0xcdcdcdff, 0xcececeff, 0xcfcfcfff, 0xd0d0d0ff, 0xd1d1d1ff, 0xd2d2d2ff, 0xd3d3d3ff, 0xd4d4d4ff, 0xd5d5d5ff, 0xd6d6d6ff, 0xd7d7d7ff,
        0xd8d8d8ff, 0xd9d9d9ff, 0xdadadaff, 0xdbdbdbff, 0xdcdcdcff, 0xddddddff, 0xdededeff, 0xdfdfdfff, 0xe0e0e0ff, 0xe1e1e1ff, 0xe2e2e2ff, 0xe3e3e3ff,
        0xe4e4e4ff, 0xe5e5e5ff, 0xe6e6e6ff, 0xe7e7e7ff, 0xe8e8e8ff, 0xe9e9e9ff, 0xeaeaeaff, 0xebebebff, 0xecececff, 0xedededff, 0xeeeeeeff, 0xefefefff,
        0xf0f0f0ff, 0xf1f1f1ff, 0xf2f2f2ff, 0xf3f3f3ff, 0xf4f4f4ff, 0xf5f5f5ff, 0xf6f6f6ff, 0xf7f7f7ff, 0xf8f8f8ff, 0xf9f9f9ff, 0xfafafaff, 0xfbfbfbff,
        0xfcfcfcff, 0xfdfdfdff, 0xfefefeff, 0xffffffff];
    exports.PuOr3 = [0x998ec3ff, 0xf7f7f7ff, 0xf1a340ff];
    exports.PuOr4 = [0x5e3c99ff, 0xb2abd2ff, 0xfdb863ff, 0xe66101ff];
    exports.PuOr5 = [0x5e3c99ff, 0xb2abd2ff, 0xf7f7f7ff, 0xfdb863ff, 0xe66101ff];
    exports.PuOr6 = [0x542788ff, 0x998ec3ff, 0xd8daebff, 0xfee0b6ff, 0xf1a340ff, 0xb35806ff];
    exports.PuOr7 = [0x542788ff, 0x998ec3ff, 0xd8daebff, 0xf7f7f7ff, 0xfee0b6ff, 0xf1a340ff, 0xb35806ff];
    exports.PuOr8 = [0x542788ff, 0x8073acff, 0xb2abd2ff, 0xd8daebff, 0xfee0b6ff, 0xfdb863ff, 0xe08214ff, 0xb35806ff];
    exports.PuOr9 = [0x542788ff, 0x8073acff, 0xb2abd2ff, 0xd8daebff, 0xf7f7f7ff, 0xfee0b6ff, 0xfdb863ff, 0xe08214ff, 0xb35806ff];
    exports.PuOr10 = [0x2d004bff, 0x542788ff, 0x8073acff, 0xb2abd2ff, 0xd8daebff, 0xfee0b6ff, 0xfdb863ff, 0xe08214ff, 0xb35806ff, 0x7f3b08ff];
    exports.PuOr11 = [0x2d004bff, 0x542788ff, 0x8073acff, 0xb2abd2ff, 0xd8daebff, 0xf7f7f7ff, 0xfee0b6ff, 0xfdb863ff, 0xe08214ff, 0xb35806ff, 0x7f3b08ff];
    exports.BrBG3 = [0x5ab4acff, 0xf5f5f5ff, 0xd8b365ff];
    exports.BrBG4 = [0x018571ff, 0x80cdc1ff, 0xdfc27dff, 0xa6611aff];
    exports.BrBG5 = [0x018571ff, 0x80cdc1ff, 0xf5f5f5ff, 0xdfc27dff, 0xa6611aff];
    exports.BrBG6 = [0x01665eff, 0x5ab4acff, 0xc7eae5ff, 0xf6e8c3ff, 0xd8b365ff, 0x8c510aff];
    exports.BrBG7 = [0x01665eff, 0x5ab4acff, 0xc7eae5ff, 0xf5f5f5ff, 0xf6e8c3ff, 0xd8b365ff, 0x8c510aff];
    exports.BrBG8 = [0x01665eff, 0x35978fff, 0x80cdc1ff, 0xc7eae5ff, 0xf6e8c3ff, 0xdfc27dff, 0xbf812dff, 0x8c510aff];
    exports.BrBG9 = [0x01665eff, 0x35978fff, 0x80cdc1ff, 0xc7eae5ff, 0xf5f5f5ff, 0xf6e8c3ff, 0xdfc27dff, 0xbf812dff, 0x8c510aff];
    exports.BrBG10 = [0x003c30ff, 0x01665eff, 0x35978fff, 0x80cdc1ff, 0xc7eae5ff, 0xf6e8c3ff, 0xdfc27dff, 0xbf812dff, 0x8c510aff, 0x543005ff];
    exports.BrBG11 = [0x003c30ff, 0x01665eff, 0x35978fff, 0x80cdc1ff, 0xc7eae5ff, 0xf5f5f5ff, 0xf6e8c3ff, 0xdfc27dff, 0xbf812dff, 0x8c510aff, 0x543005ff];
    exports.PRGn3 = [0x7fbf7bff, 0xf7f7f7ff, 0xaf8dc3ff];
    exports.PRGn4 = [0x008837ff, 0xa6dba0ff, 0xc2a5cfff, 0x7b3294ff];
    exports.PRGn5 = [0x008837ff, 0xa6dba0ff, 0xf7f7f7ff, 0xc2a5cfff, 0x7b3294ff];
    exports.PRGn6 = [0x1b7837ff, 0x7fbf7bff, 0xd9f0d3ff, 0xe7d4e8ff, 0xaf8dc3ff, 0x762a83ff];
    exports.PRGn7 = [0x1b7837ff, 0x7fbf7bff, 0xd9f0d3ff, 0xf7f7f7ff, 0xe7d4e8ff, 0xaf8dc3ff, 0x762a83ff];
    exports.PRGn8 = [0x1b7837ff, 0x5aae61ff, 0xa6dba0ff, 0xd9f0d3ff, 0xe7d4e8ff, 0xc2a5cfff, 0x9970abff, 0x762a83ff];
    exports.PRGn9 = [0x1b7837ff, 0x5aae61ff, 0xa6dba0ff, 0xd9f0d3ff, 0xf7f7f7ff, 0xe7d4e8ff, 0xc2a5cfff, 0x9970abff, 0x762a83ff];
    exports.PRGn10 = [0x00441bff, 0x1b7837ff, 0x5aae61ff, 0xa6dba0ff, 0xd9f0d3ff, 0xe7d4e8ff, 0xc2a5cfff, 0x9970abff, 0x762a83ff, 0x40004bff];
    exports.PRGn11 = [0x00441bff, 0x1b7837ff, 0x5aae61ff, 0xa6dba0ff, 0xd9f0d3ff, 0xf7f7f7ff, 0xe7d4e8ff, 0xc2a5cfff, 0x9970abff, 0x762a83ff, 0x40004bff];
    exports.PiYG3 = [0xa1d76aff, 0xf7f7f7ff, 0xe9a3c9ff];
    exports.PiYG4 = [0x4dac26ff, 0xb8e186ff, 0xf1b6daff, 0xd01c8bff];
    exports.PiYG5 = [0x4dac26ff, 0xb8e186ff, 0xf7f7f7ff, 0xf1b6daff, 0xd01c8bff];
    exports.PiYG6 = [0x4d9221ff, 0xa1d76aff, 0xe6f5d0ff, 0xfde0efff, 0xe9a3c9ff, 0xc51b7dff];
    exports.PiYG7 = [0x4d9221ff, 0xa1d76aff, 0xe6f5d0ff, 0xf7f7f7ff, 0xfde0efff, 0xe9a3c9ff, 0xc51b7dff];
    exports.PiYG8 = [0x4d9221ff, 0x7fbc41ff, 0xb8e186ff, 0xe6f5d0ff, 0xfde0efff, 0xf1b6daff, 0xde77aeff, 0xc51b7dff];
    exports.PiYG9 = [0x4d9221ff, 0x7fbc41ff, 0xb8e186ff, 0xe6f5d0ff, 0xf7f7f7ff, 0xfde0efff, 0xf1b6daff, 0xde77aeff, 0xc51b7dff];
    exports.PiYG10 = [0x276419ff, 0x4d9221ff, 0x7fbc41ff, 0xb8e186ff, 0xe6f5d0ff, 0xfde0efff, 0xf1b6daff, 0xde77aeff, 0xc51b7dff, 0x8e0152ff];
    exports.PiYG11 = [0x276419ff, 0x4d9221ff, 0x7fbc41ff, 0xb8e186ff, 0xe6f5d0ff, 0xf7f7f7ff, 0xfde0efff, 0xf1b6daff, 0xde77aeff, 0xc51b7dff, 0x8e0152ff];
    exports.RdBu3 = [0x67a9cfff, 0xf7f7f7ff, 0xef8a62ff];
    exports.RdBu4 = [0x0571b0ff, 0x92c5deff, 0xf4a582ff, 0xca0020ff];
    exports.RdBu5 = [0x0571b0ff, 0x92c5deff, 0xf7f7f7ff, 0xf4a582ff, 0xca0020ff];
    exports.RdBu6 = [0x2166acff, 0x67a9cfff, 0xd1e5f0ff, 0xfddbc7ff, 0xef8a62ff, 0xb2182bff];
    exports.RdBu7 = [0x2166acff, 0x67a9cfff, 0xd1e5f0ff, 0xf7f7f7ff, 0xfddbc7ff, 0xef8a62ff, 0xb2182bff];
    exports.RdBu8 = [0x2166acff, 0x4393c3ff, 0x92c5deff, 0xd1e5f0ff, 0xfddbc7ff, 0xf4a582ff, 0xd6604dff, 0xb2182bff];
    exports.RdBu9 = [0x2166acff, 0x4393c3ff, 0x92c5deff, 0xd1e5f0ff, 0xf7f7f7ff, 0xfddbc7ff, 0xf4a582ff, 0xd6604dff, 0xb2182bff];
    exports.RdBu10 = [0x053061ff, 0x2166acff, 0x4393c3ff, 0x92c5deff, 0xd1e5f0ff, 0xfddbc7ff, 0xf4a582ff, 0xd6604dff, 0xb2182bff, 0x67001fff];
    exports.RdBu11 = [0x053061ff, 0x2166acff, 0x4393c3ff, 0x92c5deff, 0xd1e5f0ff, 0xf7f7f7ff, 0xfddbc7ff, 0xf4a582ff, 0xd6604dff, 0xb2182bff, 0x67001fff];
    exports.RdGy3 = [0x999999ff, 0xffffffff, 0xef8a62ff];
    exports.RdGy4 = [0x404040ff, 0xbababaff, 0xf4a582ff, 0xca0020ff];
    exports.RdGy5 = [0x404040ff, 0xbababaff, 0xffffffff, 0xf4a582ff, 0xca0020ff];
    exports.RdGy6 = [0x4d4d4dff, 0x999999ff, 0xe0e0e0ff, 0xfddbc7ff, 0xef8a62ff, 0xb2182bff];
    exports.RdGy7 = [0x4d4d4dff, 0x999999ff, 0xe0e0e0ff, 0xffffffff, 0xfddbc7ff, 0xef8a62ff, 0xb2182bff];
    exports.RdGy8 = [0x4d4d4dff, 0x878787ff, 0xbababaff, 0xe0e0e0ff, 0xfddbc7ff, 0xf4a582ff, 0xd6604dff, 0xb2182bff];
    exports.RdGy9 = [0x4d4d4dff, 0x878787ff, 0xbababaff, 0xe0e0e0ff, 0xffffffff, 0xfddbc7ff, 0xf4a582ff, 0xd6604dff, 0xb2182bff];
    exports.RdGy10 = [0x1a1a1aff, 0x4d4d4dff, 0x878787ff, 0xbababaff, 0xe0e0e0ff, 0xfddbc7ff, 0xf4a582ff, 0xd6604dff, 0xb2182bff, 0x67001fff];
    exports.RdGy11 = [0x1a1a1aff, 0x4d4d4dff, 0x878787ff, 0xbababaff, 0xe0e0e0ff, 0xffffffff, 0xfddbc7ff, 0xf4a582ff, 0xd6604dff, 0xb2182bff, 0x67001fff];
    exports.RdYlBu3 = [0x91bfdbff, 0xffffbfff, 0xfc8d59ff];
    exports.RdYlBu4 = [0x2c7bb6ff, 0xabd9e9ff, 0xfdae61ff, 0xd7191cff];
    exports.RdYlBu5 = [0x2c7bb6ff, 0xabd9e9ff, 0xffffbfff, 0xfdae61ff, 0xd7191cff];
    exports.RdYlBu6 = [0x4575b4ff, 0x91bfdbff, 0xe0f3f8ff, 0xfee090ff, 0xfc8d59ff, 0xd73027ff];
    exports.RdYlBu7 = [0x4575b4ff, 0x91bfdbff, 0xe0f3f8ff, 0xffffbfff, 0xfee090ff, 0xfc8d59ff, 0xd73027ff];
    exports.RdYlBu8 = [0x4575b4ff, 0x74add1ff, 0xabd9e9ff, 0xe0f3f8ff, 0xfee090ff, 0xfdae61ff, 0xf46d43ff, 0xd73027ff];
    exports.RdYlBu9 = [0x4575b4ff, 0x74add1ff, 0xabd9e9ff, 0xe0f3f8ff, 0xffffbfff, 0xfee090ff, 0xfdae61ff, 0xf46d43ff, 0xd73027ff];
    exports.RdYlBu10 = [0x313695ff, 0x4575b4ff, 0x74add1ff, 0xabd9e9ff, 0xe0f3f8ff, 0xfee090ff, 0xfdae61ff, 0xf46d43ff, 0xd73027ff, 0xa50026ff];
    exports.RdYlBu11 = [0x313695ff, 0x4575b4ff, 0x74add1ff, 0xabd9e9ff, 0xe0f3f8ff, 0xffffbfff, 0xfee090ff, 0xfdae61ff, 0xf46d43ff, 0xd73027ff, 0xa50026ff];
    exports.Spectral3 = [0x99d594ff, 0xffffbfff, 0xfc8d59ff];
    exports.Spectral4 = [0x2b83baff, 0xabdda4ff, 0xfdae61ff, 0xd7191cff];
    exports.Spectral5 = [0x2b83baff, 0xabdda4ff, 0xffffbfff, 0xfdae61ff, 0xd7191cff];
    exports.Spectral6 = [0x3288bdff, 0x99d594ff, 0xe6f598ff, 0xfee08bff, 0xfc8d59ff, 0xd53e4fff];
    exports.Spectral7 = [0x3288bdff, 0x99d594ff, 0xe6f598ff, 0xffffbfff, 0xfee08bff, 0xfc8d59ff, 0xd53e4fff];
    exports.Spectral8 = [0x3288bdff, 0x66c2a5ff, 0xabdda4ff, 0xe6f598ff, 0xfee08bff, 0xfdae61ff, 0xf46d43ff, 0xd53e4fff];
    exports.Spectral9 = [0x3288bdff, 0x66c2a5ff, 0xabdda4ff, 0xe6f598ff, 0xffffbfff, 0xfee08bff, 0xfdae61ff, 0xf46d43ff, 0xd53e4fff];
    exports.Spectral10 = [0x5e4fa2ff, 0x3288bdff, 0x66c2a5ff, 0xabdda4ff, 0xe6f598ff, 0xfee08bff, 0xfdae61ff, 0xf46d43ff, 0xd53e4fff, 0x9e0142ff];
    exports.Spectral11 = [0x5e4fa2ff, 0x3288bdff, 0x66c2a5ff, 0xabdda4ff, 0xe6f598ff, 0xffffbfff, 0xfee08bff, 0xfdae61ff, 0xf46d43ff, 0xd53e4fff, 0x9e0142ff];
    exports.RdYlGn3 = [0x91cf60ff, 0xffffbfff, 0xfc8d59ff];
    exports.RdYlGn4 = [0x1a9641ff, 0xa6d96aff, 0xfdae61ff, 0xd7191cff];
    exports.RdYlGn5 = [0x1a9641ff, 0xa6d96aff, 0xffffbfff, 0xfdae61ff, 0xd7191cff];
    exports.RdYlGn6 = [0x1a9850ff, 0x91cf60ff, 0xd9ef8bff, 0xfee08bff, 0xfc8d59ff, 0xd73027ff];
    exports.RdYlGn7 = [0x1a9850ff, 0x91cf60ff, 0xd9ef8bff, 0xffffbfff, 0xfee08bff, 0xfc8d59ff, 0xd73027ff];
    exports.RdYlGn8 = [0x1a9850ff, 0x66bd63ff, 0xa6d96aff, 0xd9ef8bff, 0xfee08bff, 0xfdae61ff, 0xf46d43ff, 0xd73027ff];
    exports.RdYlGn9 = [0x1a9850ff, 0x66bd63ff, 0xa6d96aff, 0xd9ef8bff, 0xffffbfff, 0xfee08bff, 0xfdae61ff, 0xf46d43ff, 0xd73027ff];
    exports.RdYlGn10 = [0x006837ff, 0x1a9850ff, 0x66bd63ff, 0xa6d96aff, 0xd9ef8bff, 0xfee08bff, 0xfdae61ff, 0xf46d43ff, 0xd73027ff, 0xa50026ff];
    exports.RdYlGn11 = [0x006837ff, 0x1a9850ff, 0x66bd63ff, 0xa6d96aff, 0xd9ef8bff, 0xffffbfff, 0xfee08bff, 0xfdae61ff, 0xf46d43ff, 0xd73027ff, 0xa50026ff];
    exports.Inferno3 = [0x000003ff, 0xba3655ff, 0xfcfea4ff];
    exports.Inferno4 = [0x000003ff, 0x781c6dff, 0xed6825ff, 0xfcfea4ff];
    exports.Inferno5 = [0x000003ff, 0x550f6dff, 0xba3655ff, 0xf98c09ff, 0xfcfea4ff];
    exports.Inferno6 = [0x000003ff, 0x410967ff, 0x932567ff, 0xdc5039ff, 0xfba40aff, 0xfcfea4ff];
    exports.Inferno7 = [0x000003ff, 0x32095dff, 0x781c6dff, 0xba3655ff, 0xed6825ff, 0xfbb318ff, 0xfcfea4ff];
    exports.Inferno8 = [0x000003ff, 0x270b52ff, 0x63146eff, 0x9e2963ff, 0xd24742ff, 0xf57c15ff, 0xfabf25ff, 0xfcfea4ff];
    exports.Inferno9 = [0x000003ff, 0x1f0c47ff, 0x550f6dff, 0x88216aff, 0xba3655ff, 0xe35832ff, 0xf98c09ff, 0xf8c931ff, 0xfcfea4ff];
    exports.Inferno10 = [0x000003ff, 0x1a0b40ff, 0x4a0b6aff, 0x781c6dff, 0xa42c60ff, 0xcd4247ff, 0xed6825ff, 0xfb9906ff, 0xf7cf3aff, 0xfcfea4ff];
    exports.Inferno11 = [0x000003ff, 0x160b39ff, 0x410967ff, 0x6a176eff, 0x932567ff, 0xba3655ff, 0xdc5039ff, 0xf2751aff, 0xfba40aff, 0xf6d542ff, 0xfcfea4ff];
    exports.Inferno256 = [0x000003ff, 0x000004ff, 0x000006ff, 0x010007ff, 0x010109ff, 0x01010bff, 0x02010eff, 0x020210ff, 0x030212ff, 0x040314ff, 0x040316ff, 0x050418ff,
        0x06041bff, 0x07051dff, 0x08061fff, 0x090621ff, 0x0a0723ff, 0x0b0726ff, 0x0d0828ff, 0x0e082aff, 0x0f092dff, 0x10092fff, 0x120a32ff, 0x130a34ff,
        0x140b36ff, 0x160b39ff, 0x170b3bff, 0x190b3eff, 0x1a0b40ff, 0x1c0c43ff, 0x1d0c45ff, 0x1f0c47ff, 0x200c4aff, 0x220b4cff, 0x240b4eff, 0x260b50ff,
        0x270b52ff, 0x290b54ff, 0x2b0a56ff, 0x2d0a58ff, 0x2e0a5aff, 0x300a5cff, 0x32095dff, 0x34095fff, 0x350960ff, 0x370961ff, 0x390962ff, 0x3b0964ff,
        0x3c0965ff, 0x3e0966ff, 0x400966ff, 0x410967ff, 0x430a68ff, 0x450a69ff, 0x460a69ff, 0x480b6aff, 0x4a0b6aff, 0x4b0c6bff, 0x4d0c6bff, 0x4f0d6cff,
        0x500d6cff, 0x520e6cff, 0x530e6dff, 0x550f6dff, 0x570f6dff, 0x58106dff, 0x5a116dff, 0x5b116eff, 0x5d126eff, 0x5f126eff, 0x60136eff, 0x62146eff,
        0x63146eff, 0x65156eff, 0x66156eff, 0x68166eff, 0x6a176eff, 0x6b176eff, 0x6d186eff, 0x6e186eff, 0x70196eff, 0x72196dff, 0x731a6dff, 0x751b6dff,
        0x761b6dff, 0x781c6dff, 0x7a1c6dff, 0x7b1d6cff, 0x7d1d6cff, 0x7e1e6cff, 0x801f6bff, 0x811f6bff, 0x83206bff, 0x85206aff, 0x86216aff, 0x88216aff,
        0x892269ff, 0x8b2269ff, 0x8d2369ff, 0x8e2468ff, 0x902468ff, 0x912567ff, 0x932567ff, 0x952666ff, 0x962666ff, 0x982765ff, 0x992864ff, 0x9b2864ff,
        0x9c2963ff, 0x9e2963ff, 0xa02a62ff, 0xa12b61ff, 0xa32b61ff, 0xa42c60ff, 0xa62c5fff, 0xa72d5fff, 0xa92e5eff, 0xab2e5dff, 0xac2f5cff, 0xae305bff,
        0xaf315bff, 0xb1315aff, 0xb23259ff, 0xb43358ff, 0xb53357ff, 0xb73456ff, 0xb83556ff, 0xba3655ff, 0xbb3754ff, 0xbd3753ff, 0xbe3852ff, 0xbf3951ff,
        0xc13a50ff, 0xc23b4fff, 0xc43c4eff, 0xc53d4dff, 0xc73e4cff, 0xc83e4bff, 0xc93f4aff, 0xcb4049ff, 0xcc4148ff, 0xcd4247ff, 0xcf4446ff, 0xd04544ff,
        0xd14643ff, 0xd24742ff, 0xd44841ff, 0xd54940ff, 0xd64a3fff, 0xd74b3eff, 0xd94d3dff, 0xda4e3bff, 0xdb4f3aff, 0xdc5039ff, 0xdd5238ff, 0xde5337ff,
        0xdf5436ff, 0xe05634ff, 0xe25733ff, 0xe35832ff, 0xe45a31ff, 0xe55b30ff, 0xe65c2eff, 0xe65e2dff, 0xe75f2cff, 0xe8612bff, 0xe9622aff, 0xea6428ff,
        0xeb6527ff, 0xec6726ff, 0xed6825ff, 0xed6a23ff, 0xee6c22ff, 0xef6d21ff, 0xf06f1fff, 0xf0701eff, 0xf1721dff, 0xf2741cff, 0xf2751aff, 0xf37719ff,
        0xf37918ff, 0xf47a16ff, 0xf57c15ff, 0xf57e14ff, 0xf68012ff, 0xf68111ff, 0xf78310ff, 0xf7850eff, 0xf8870dff, 0xf8880cff, 0xf88a0bff, 0xf98c09ff,
        0xf98e08ff, 0xf99008ff, 0xfa9107ff, 0xfa9306ff, 0xfa9506ff, 0xfa9706ff, 0xfb9906ff, 0xfb9b06ff, 0xfb9d06ff, 0xfb9e07ff, 0xfba007ff, 0xfba208ff,
        0xfba40aff, 0xfba60bff, 0xfba80dff, 0xfbaa0eff, 0xfbac10ff, 0xfbae12ff, 0xfbb014ff, 0xfbb116ff, 0xfbb318ff, 0xfbb51aff, 0xfbb71cff, 0xfbb91eff,
        0xfabb21ff, 0xfabd23ff, 0xfabf25ff, 0xfac128ff, 0xf9c32aff, 0xf9c52cff, 0xf9c72fff, 0xf8c931ff, 0xf8cb34ff, 0xf8cd37ff, 0xf7cf3aff, 0xf7d13cff,
        0xf6d33fff, 0xf6d542ff, 0xf5d745ff, 0xf5d948ff, 0xf4db4bff, 0xf4dc4fff, 0xf3de52ff, 0xf3e056ff, 0xf3e259ff, 0xf2e45dff, 0xf2e660ff, 0xf1e864ff,
        0xf1e968ff, 0xf1eb6cff, 0xf1ed70ff, 0xf1ee74ff, 0xf1f079ff, 0xf1f27dff, 0xf2f381ff, 0xf2f485ff, 0xf3f689ff, 0xf4f78dff, 0xf5f891ff, 0xf6fa95ff,
        0xf7fb99ff, 0xf9fc9dff, 0xfafda0ff, 0xfcfea4ff];
    exports.Magma3 = [0x000003ff, 0xb53679ff, 0xfbfcbfff];
    exports.Magma4 = [0x000003ff, 0x711f81ff, 0xf0605dff, 0xfbfcbfff];
    exports.Magma5 = [0x000003ff, 0x4f117bff, 0xb53679ff, 0xfb8660ff, 0xfbfcbfff];
    exports.Magma6 = [0x000003ff, 0x3b0f6fff, 0x8c2980ff, 0xdd4968ff, 0xfd9f6cff, 0xfbfcbfff];
    exports.Magma7 = [0x000003ff, 0x2b115eff, 0x711f81ff, 0xb53679ff, 0xf0605dff, 0xfeae76ff, 0xfbfcbfff];
    exports.Magma8 = [0x000003ff, 0x221150ff, 0x5d177eff, 0x972c7fff, 0xd1426eff, 0xf8755cff, 0xfeb97fff, 0xfbfcbfff];
    exports.Magma9 = [0x000003ff, 0x1b1044ff, 0x4f117bff, 0x812581ff, 0xb53679ff, 0xe55063ff, 0xfb8660ff, 0xfec286ff, 0xfbfcbfff];
    exports.Magma10 = [0x000003ff, 0x170f3cff, 0x430f75ff, 0x711f81ff, 0x9e2e7eff, 0xcb3e71ff, 0xf0605dff, 0xfc9366ff, 0xfec78bff, 0xfbfcbfff];
    exports.Magma11 = [0x000003ff, 0x140d35ff, 0x3b0f6fff, 0x63197fff, 0x8c2980ff, 0xb53679ff, 0xdd4968ff, 0xf66e5bff, 0xfd9f6cff, 0xfdcd90ff, 0xfbfcbfff];
    exports.Magma256 = [0x000003ff, 0x000004ff, 0x000006ff, 0x010007ff, 0x010109ff, 0x01010bff, 0x02020dff, 0x02020fff, 0x030311ff, 0x040313ff, 0x040415ff, 0x050417ff,
        0x060519ff, 0x07051bff, 0x08061dff, 0x09071fff, 0x0a0722ff, 0x0b0824ff, 0x0c0926ff, 0x0d0a28ff, 0x0e0a2aff, 0x0f0b2cff, 0x100c2fff, 0x110c31ff,
        0x120d33ff, 0x140d35ff, 0x150e38ff, 0x160e3aff, 0x170f3cff, 0x180f3fff, 0x1a1041ff, 0x1b1044ff, 0x1c1046ff, 0x1e1049ff, 0x1f114bff, 0x20114dff,
        0x221150ff, 0x231152ff, 0x251155ff, 0x261157ff, 0x281159ff, 0x2a115cff, 0x2b115eff, 0x2d1060ff, 0x2f1062ff, 0x301065ff, 0x321067ff, 0x341068ff,
        0x350f6aff, 0x370f6cff, 0x390f6eff, 0x3b0f6fff, 0x3c0f71ff, 0x3e0f72ff, 0x400f73ff, 0x420f74ff, 0x430f75ff, 0x450f76ff, 0x470f77ff, 0x481078ff,
        0x4a1079ff, 0x4b1079ff, 0x4d117aff, 0x4f117bff, 0x50127bff, 0x52127cff, 0x53137cff, 0x55137dff, 0x57147dff, 0x58157eff, 0x5a157eff, 0x5b167eff,
        0x5d177eff, 0x5e177fff, 0x60187fff, 0x61187fff, 0x63197fff, 0x651a80ff, 0x661a80ff, 0x681b80ff, 0x691c80ff, 0x6b1c80ff, 0x6c1d80ff, 0x6e1e81ff,
        0x6f1e81ff, 0x711f81ff, 0x731f81ff, 0x742081ff, 0x762181ff, 0x772181ff, 0x792281ff, 0x7a2281ff, 0x7c2381ff, 0x7e2481ff, 0x7f2481ff, 0x812581ff,
        0x822581ff, 0x842681ff, 0x852681ff, 0x872781ff, 0x892881ff, 0x8a2881ff, 0x8c2980ff, 0x8d2980ff, 0x8f2a80ff, 0x912a80ff, 0x922b80ff, 0x942b80ff,
        0x952c80ff, 0x972c7fff, 0x992d7fff, 0x9a2d7fff, 0x9c2e7fff, 0x9e2e7eff, 0x9f2f7eff, 0xa12f7eff, 0xa3307eff, 0xa4307dff, 0xa6317dff, 0xa7317dff,
        0xa9327cff, 0xab337cff, 0xac337bff, 0xae347bff, 0xb0347bff, 0xb1357aff, 0xb3357aff, 0xb53679ff, 0xb63679ff, 0xb83778ff, 0xb93778ff, 0xbb3877ff,
        0xbd3977ff, 0xbe3976ff, 0xc03a75ff, 0xc23a75ff, 0xc33b74ff, 0xc53c74ff, 0xc63c73ff, 0xc83d72ff, 0xca3e72ff, 0xcb3e71ff, 0xcd3f70ff, 0xce4070ff,
        0xd0416fff, 0xd1426eff, 0xd3426dff, 0xd4436dff, 0xd6446cff, 0xd7456bff, 0xd9466aff, 0xda4769ff, 0xdc4869ff, 0xdd4968ff, 0xde4a67ff, 0xe04b66ff,
        0xe14c66ff, 0xe24d65ff, 0xe44e64ff, 0xe55063ff, 0xe65162ff, 0xe75262ff, 0xe85461ff, 0xea5560ff, 0xeb5660ff, 0xec585fff, 0xed595fff, 0xee5b5eff,
        0xee5d5dff, 0xef5e5dff, 0xf0605dff, 0xf1615cff, 0xf2635cff, 0xf3655cff, 0xf3675bff, 0xf4685bff, 0xf56a5bff, 0xf56c5bff, 0xf66e5bff, 0xf6705bff,
        0xf7715bff, 0xf7735cff, 0xf8755cff, 0xf8775cff, 0xf9795cff, 0xf97b5dff, 0xf97d5dff, 0xfa7f5eff, 0xfa805eff, 0xfa825fff, 0xfb8460ff, 0xfb8660ff,
        0xfb8861ff, 0xfb8a62ff, 0xfc8c63ff, 0xfc8e63ff, 0xfc9064ff, 0xfc9265ff, 0xfc9366ff, 0xfd9567ff, 0xfd9768ff, 0xfd9969ff, 0xfd9b6aff, 0xfd9d6bff,
        0xfd9f6cff, 0xfda16eff, 0xfda26fff, 0xfda470ff, 0xfea671ff, 0xfea873ff, 0xfeaa74ff, 0xfeac75ff, 0xfeae76ff, 0xfeaf78ff, 0xfeb179ff, 0xfeb37bff,
        0xfeb57cff, 0xfeb77dff, 0xfeb97fff, 0xfebb80ff, 0xfebc82ff, 0xfebe83ff, 0xfec085ff, 0xfec286ff, 0xfec488ff, 0xfec689ff, 0xfec78bff, 0xfec98dff,
        0xfecb8eff, 0xfdcd90ff, 0xfdcf92ff, 0xfdd193ff, 0xfdd295ff, 0xfdd497ff, 0xfdd698ff, 0xfdd89aff, 0xfdda9cff, 0xfddc9dff, 0xfddd9fff, 0xfddfa1ff,
        0xfde1a3ff, 0xfce3a5ff, 0xfce5a6ff, 0xfce6a8ff, 0xfce8aaff, 0xfceaacff, 0xfcecaeff, 0xfceeb0ff, 0xfcf0b1ff, 0xfcf1b3ff, 0xfcf3b5ff, 0xfcf5b7ff,
        0xfbf7b9ff, 0xfbf9bbff, 0xfbfabdff, 0xfbfcbfff];
    exports.Plasma3 = [0x0c0786ff, 0xca4678ff, 0xeff821ff];
    exports.Plasma4 = [0x0c0786ff, 0x9b179eff, 0xec7853ff, 0xeff821ff];
    exports.Plasma5 = [0x0c0786ff, 0x7c02a7ff, 0xca4678ff, 0xf79341ff, 0xeff821ff];
    exports.Plasma6 = [0x0c0786ff, 0x6a00a7ff, 0xb02a8fff, 0xe06461ff, 0xfca635ff, 0xeff821ff];
    exports.Plasma7 = [0x0c0786ff, 0x5c00a5ff, 0x9b179eff, 0xca4678ff, 0xec7853ff, 0xfdb22fff, 0xeff821ff];
    exports.Plasma8 = [0x0c0786ff, 0x5201a3ff, 0x8908a5ff, 0xb83289ff, 0xda5a68ff, 0xf38748ff, 0xfdbb2bff, 0xeff821ff];
    exports.Plasma9 = [0x0c0786ff, 0x4a02a0ff, 0x7c02a7ff, 0xa82296ff, 0xca4678ff, 0xe56b5cff, 0xf79341ff, 0xfdc328ff, 0xeff821ff];
    exports.Plasma10 = [0x0c0786ff, 0x45039eff, 0x7200a8ff, 0x9b179eff, 0xbc3685ff, 0xd7566cff, 0xec7853ff, 0xfa9d3aff, 0xfcc726ff, 0xeff821ff];
    exports.Plasma11 = [0x0c0786ff, 0x40039cff, 0x6a00a7ff, 0x8f0da3ff, 0xb02a8fff, 0xca4678ff, 0xe06461ff, 0xf1824cff, 0xfca635ff, 0xfccc25ff, 0xeff821ff];
    exports.Plasma256 = [0x0c0786ff, 0x100787ff, 0x130689ff, 0x15068aff, 0x18068bff, 0x1b068cff, 0x1d068dff, 0x1f058eff, 0x21058fff, 0x230590ff, 0x250591ff, 0x270592ff,
        0x290593ff, 0x2b0594ff, 0x2d0494ff, 0x2f0495ff, 0x310496ff, 0x330497ff, 0x340498ff, 0x360498ff, 0x380499ff, 0x3a049aff, 0x3b039aff, 0x3d039bff,
        0x3f039cff, 0x40039cff, 0x42039dff, 0x44039eff, 0x45039eff, 0x47029fff, 0x49029fff, 0x4a02a0ff, 0x4c02a1ff, 0x4e02a1ff, 0x4f02a2ff, 0x5101a2ff,
        0x5201a3ff, 0x5401a3ff, 0x5601a3ff, 0x5701a4ff, 0x5901a4ff, 0x5a00a5ff, 0x5c00a5ff, 0x5e00a5ff, 0x5f00a6ff, 0x6100a6ff, 0x6200a6ff, 0x6400a7ff,
        0x6500a7ff, 0x6700a7ff, 0x6800a7ff, 0x6a00a7ff, 0x6c00a8ff, 0x6d00a8ff, 0x6f00a8ff, 0x7000a8ff, 0x7200a8ff, 0x7300a8ff, 0x7500a8ff, 0x7601a8ff,
        0x7801a8ff, 0x7901a8ff, 0x7b02a8ff, 0x7c02a7ff, 0x7e03a7ff, 0x7f03a7ff, 0x8104a7ff, 0x8204a7ff, 0x8405a6ff, 0x8506a6ff, 0x8607a6ff, 0x8807a5ff,
        0x8908a5ff, 0x8b09a4ff, 0x8c0aa4ff, 0x8e0ca4ff, 0x8f0da3ff, 0x900ea3ff, 0x920fa2ff, 0x9310a1ff, 0x9511a1ff, 0x9612a0ff, 0x9713a0ff, 0x99149fff,
        0x9a159eff, 0x9b179eff, 0x9d189dff, 0x9e199cff, 0x9f1a9bff, 0xa01b9bff, 0xa21c9aff, 0xa31d99ff, 0xa41e98ff, 0xa51f97ff, 0xa72197ff, 0xa82296ff,
        0xa92395ff, 0xaa2494ff, 0xac2593ff, 0xad2692ff, 0xae2791ff, 0xaf2890ff, 0xb02a8fff, 0xb12b8fff, 0xb22c8eff, 0xb42d8dff, 0xb52e8cff, 0xb62f8bff,
        0xb7308aff, 0xb83289ff, 0xb93388ff, 0xba3487ff, 0xbb3586ff, 0xbc3685ff, 0xbd3784ff, 0xbe3883ff, 0xbf3982ff, 0xc03b81ff, 0xc13c80ff, 0xc23d80ff,
        0xc33e7fff, 0xc43f7eff, 0xc5407dff, 0xc6417cff, 0xc7427bff, 0xc8447aff, 0xc94579ff, 0xca4678ff, 0xcb4777ff, 0xcc4876ff, 0xcd4975ff, 0xce4a75ff,
        0xcf4b74ff, 0xd04d73ff, 0xd14e72ff, 0xd14f71ff, 0xd25070ff, 0xd3516fff, 0xd4526eff, 0xd5536dff, 0xd6556dff, 0xd7566cff, 0xd7576bff, 0xd8586aff,
        0xd95969ff, 0xda5a68ff, 0xdb5b67ff, 0xdc5d66ff, 0xdc5e66ff, 0xdd5f65ff, 0xde6064ff, 0xdf6163ff, 0xdf6262ff, 0xe06461ff, 0xe16560ff, 0xe26660ff,
        0xe3675fff, 0xe3685eff, 0xe46a5dff, 0xe56b5cff, 0xe56c5bff, 0xe66d5aff, 0xe76e5aff, 0xe87059ff, 0xe87158ff, 0xe97257ff, 0xea7356ff, 0xea7455ff,
        0xeb7654ff, 0xec7754ff, 0xec7853ff, 0xed7952ff, 0xed7b51ff, 0xee7c50ff, 0xef7d4fff, 0xef7e4eff, 0xf0804dff, 0xf0814dff, 0xf1824cff, 0xf2844bff,
        0xf2854aff, 0xf38649ff, 0xf38748ff, 0xf48947ff, 0xf48a47ff, 0xf58b46ff, 0xf58d45ff, 0xf68e44ff, 0xf68f43ff, 0xf69142ff, 0xf79241ff, 0xf79341ff,
        0xf89540ff, 0xf8963fff, 0xf8983eff, 0xf9993dff, 0xf99a3cff, 0xfa9c3bff, 0xfa9d3aff, 0xfa9f3aff, 0xfaa039ff, 0xfba238ff, 0xfba337ff, 0xfba436ff,
        0xfca635ff, 0xfca735ff, 0xfca934ff, 0xfcaa33ff, 0xfcac32ff, 0xfcad31ff, 0xfdaf31ff, 0xfdb030ff, 0xfdb22fff, 0xfdb32eff, 0xfdb52dff, 0xfdb62dff,
        0xfdb82cff, 0xfdb92bff, 0xfdbb2bff, 0xfdbc2aff, 0xfdbe29ff, 0xfdc029ff, 0xfdc128ff, 0xfdc328ff, 0xfdc427ff, 0xfdc626ff, 0xfcc726ff, 0xfcc926ff,
        0xfccb25ff, 0xfccc25ff, 0xfcce25ff, 0xfbd024ff, 0xfbd124ff, 0xfbd324ff, 0xfad524ff, 0xfad624ff, 0xfad824ff, 0xf9d924ff, 0xf9db24ff, 0xf8dd24ff,
        0xf8df24ff, 0xf7e024ff, 0xf7e225ff, 0xf6e425ff, 0xf6e525ff, 0xf5e726ff, 0xf5e926ff, 0xf4ea26ff, 0xf3ec26ff, 0xf3ee26ff, 0xf2f026ff, 0xf2f126ff,
        0xf1f326ff, 0xf0f525ff, 0xf0f623ff, 0xeff821ff];
    exports.Viridis3 = [0x440154ff, 0x208f8cff, 0xfde724ff];
    exports.Viridis4 = [0x440154ff, 0x30678dff, 0x35b778ff, 0xfde724ff];
    exports.Viridis5 = [0x440154ff, 0x3b518aff, 0x208f8cff, 0x5bc862ff, 0xfde724ff];
    exports.Viridis6 = [0x440154ff, 0x404387ff, 0x29788eff, 0x22a784ff, 0x79d151ff, 0xfde724ff];
    exports.Viridis7 = [0x440154ff, 0x443982ff, 0x30678dff, 0x208f8cff, 0x35b778ff, 0x8dd644ff, 0xfde724ff];
    exports.Viridis8 = [0x440154ff, 0x46317eff, 0x365a8cff, 0x277e8eff, 0x1ea087ff, 0x49c16dff, 0x9dd93aff, 0xfde724ff];
    exports.Viridis9 = [0x440154ff, 0x472b7aff, 0x3b518aff, 0x2c718eff, 0x208f8cff, 0x27ad80ff, 0x5bc862ff, 0xaadb32ff, 0xfde724ff];
    exports.Viridis10 = [0x440154ff, 0x472777ff, 0x3e4989ff, 0x30678dff, 0x25828eff, 0x1e9c89ff, 0x35b778ff, 0x6bcd59ff, 0xb2dd2cff, 0xfde724ff];
    exports.Viridis11 = [0x440154ff, 0x482374ff, 0x404387ff, 0x345e8dff, 0x29788eff, 0x208f8cff, 0x22a784ff, 0x42be71ff, 0x79d151ff, 0xbade27ff, 0xfde724ff];
    exports.Viridis256 = [0x440154ff, 0x440255ff, 0x440357ff, 0x450558ff, 0x45065aff, 0x45085bff, 0x46095cff, 0x460b5eff, 0x460c5fff, 0x460e61ff, 0x470f62ff, 0x471163ff,
        0x471265ff, 0x471466ff, 0x471567ff, 0x471669ff, 0x47186aff, 0x48196bff, 0x481a6cff, 0x481c6eff, 0x481d6fff, 0x481e70ff, 0x482071ff, 0x482172ff,
        0x482273ff, 0x482374ff, 0x472575ff, 0x472676ff, 0x472777ff, 0x472878ff, 0x472a79ff, 0x472b7aff, 0x472c7bff, 0x462d7cff, 0x462f7cff, 0x46307dff,
        0x46317eff, 0x45327fff, 0x45347fff, 0x453580ff, 0x453681ff, 0x443781ff, 0x443982ff, 0x433a83ff, 0x433b83ff, 0x433c84ff, 0x423d84ff, 0x423e85ff,
        0x424085ff, 0x414186ff, 0x414286ff, 0x404387ff, 0x404487ff, 0x3f4587ff, 0x3f4788ff, 0x3e4888ff, 0x3e4989ff, 0x3d4a89ff, 0x3d4b89ff, 0x3d4c89ff,
        0x3c4d8aff, 0x3c4e8aff, 0x3b508aff, 0x3b518aff, 0x3a528bff, 0x3a538bff, 0x39548bff, 0x39558bff, 0x38568bff, 0x38578cff, 0x37588cff, 0x37598cff,
        0x365a8cff, 0x365b8cff, 0x355c8cff, 0x355d8cff, 0x345e8dff, 0x345f8dff, 0x33608dff, 0x33618dff, 0x32628dff, 0x32638dff, 0x31648dff, 0x31658dff,
        0x31668dff, 0x30678dff, 0x30688dff, 0x2f698dff, 0x2f6a8dff, 0x2e6b8eff, 0x2e6c8eff, 0x2e6d8eff, 0x2d6e8eff, 0x2d6f8eff, 0x2c708eff, 0x2c718eff,
        0x2c728eff, 0x2b738eff, 0x2b748eff, 0x2a758eff, 0x2a768eff, 0x2a778eff, 0x29788eff, 0x29798eff, 0x287a8eff, 0x287a8eff, 0x287b8eff, 0x277c8eff,
        0x277d8eff, 0x277e8eff, 0x267f8eff, 0x26808eff, 0x26818eff, 0x25828eff, 0x25838dff, 0x24848dff, 0x24858dff, 0x24868dff, 0x23878dff, 0x23888dff,
        0x23898dff, 0x22898dff, 0x228a8dff, 0x228b8dff, 0x218c8dff, 0x218d8cff, 0x218e8cff, 0x208f8cff, 0x20908cff, 0x20918cff, 0x1f928cff, 0x1f938bff,
        0x1f948bff, 0x1f958bff, 0x1f968bff, 0x1e978aff, 0x1e988aff, 0x1e998aff, 0x1e998aff, 0x1e9a89ff, 0x1e9b89ff, 0x1e9c89ff, 0x1e9d88ff, 0x1e9e88ff,
        0x1e9f88ff, 0x1ea087ff, 0x1fa187ff, 0x1fa286ff, 0x1fa386ff, 0x20a485ff, 0x20a585ff, 0x21a685ff, 0x21a784ff, 0x22a784ff, 0x23a883ff, 0x23a982ff,
        0x24aa82ff, 0x25ab81ff, 0x26ac81ff, 0x27ad80ff, 0x28ae7fff, 0x29af7fff, 0x2ab07eff, 0x2bb17dff, 0x2cb17dff, 0x2eb27cff, 0x2fb37bff, 0x30b47aff,
        0x32b57aff, 0x33b679ff, 0x35b778ff, 0x36b877ff, 0x38b976ff, 0x39b976ff, 0x3bba75ff, 0x3dbb74ff, 0x3ebc73ff, 0x40bd72ff, 0x42be71ff, 0x44be70ff,
        0x45bf6fff, 0x47c06eff, 0x49c16dff, 0x4bc26cff, 0x4dc26bff, 0x4fc369ff, 0x51c468ff, 0x53c567ff, 0x55c666ff, 0x57c665ff, 0x59c764ff, 0x5bc862ff,
        0x5ec961ff, 0x60c960ff, 0x62ca5fff, 0x64cb5dff, 0x67cc5cff, 0x69cc5bff, 0x6bcd59ff, 0x6dce58ff, 0x70ce56ff, 0x72cf55ff, 0x74d054ff, 0x77d052ff,
        0x79d151ff, 0x7cd24fff, 0x7ed24eff, 0x81d34cff, 0x83d34bff, 0x86d449ff, 0x88d547ff, 0x8bd546ff, 0x8dd644ff, 0x90d643ff, 0x92d741ff, 0x95d73fff,
        0x97d83eff, 0x9ad83cff, 0x9dd93aff, 0x9fd938ff, 0xa2da37ff, 0xa5da35ff, 0xa7db33ff, 0xaadb32ff, 0xaddc30ff, 0xafdc2eff, 0xb2dd2cff, 0xb5dd2bff,
        0xb7dd29ff, 0xbade27ff, 0xbdde26ff, 0xbfdf24ff, 0xc2df22ff, 0xc5df21ff, 0xc7e01fff, 0xcae01eff, 0xcde01dff, 0xcfe11cff, 0xd2e11bff, 0xd4e11aff,
        0xd7e219ff, 0xdae218ff, 0xdce218ff, 0xdfe318ff, 0xe1e318ff, 0xe4e318ff, 0xe7e419ff, 0xe9e419ff, 0xece41aff, 0xeee51bff, 0xf1e51cff, 0xf3e51eff,
        0xf6e61fff, 0xf8e621ff, 0xfae622ff, 0xfde724ff];
    exports.Cividis3 = [0x00204cff, 0x7b7b78ff, 0xffe945];
    exports.Cividis4 = [0x00204cff, 0x565c6cff, 0xa69c75ff, 0xffe945];
    exports.Cividis5 = [0x00204cff, 0x404c6bff, 0x7b7b78ff, 0xbcae6eff, 0xffe945];
    exports.Cividis6 = [0x00204cff, 0x31446bff, 0x666870ff, 0x958f78ff, 0xcab969ff, 0xffe945];
    exports.Cividis7 = [0x00204cff, 0x223d6cff, 0x565c6cff, 0x7b7b78ff, 0xa69c75ff, 0xd3c065ff, 0xffe945];
    exports.Cividis8 = [0x00204cff, 0x15396dff, 0x49536bff, 0x6c6d72ff, 0x8d8878ff, 0xb2a672ff, 0xd9c661ff, 0xffe945];
    exports.Cividis9 = [0x00204cff, 0x01356eff, 0x404c6bff, 0x5f636eff, 0x7b7b78ff, 0x9b9377ff, 0xbcae6eff, 0xdfcb5dff, 0xffe945];
    exports.Cividis10 = [0x00204cff, 0x00336eff, 0x37476bff, 0x565c6cff, 0x6f7073ff, 0x898578ff, 0xa69c75ff, 0xc3b46cff, 0xe3cd5bff, 0xffe945];
    exports.Cividis11 = [0x00204cff, 0x00316fff, 0x31446bff, 0x4d556bff, 0x666870ff, 0x7b7b78ff, 0x958f78ff, 0xaea373ff, 0xcab969ff, 0xe6d059ff, 0xffe945];
    exports.Cividis256 = [
        0x00204cff, 0x00204eff, 0x002150ff, 0x002251ff, 0x002353ff, 0x002355ff, 0x002456ff, 0x002558ff,
        0x00265aff, 0x00265bff, 0x00275dff, 0x00285fff, 0x002861ff, 0x002963ff, 0x002a64ff, 0x002a66ff,
        0x002b68ff, 0x002c6aff, 0x002d6cff, 0x002d6dff, 0x002e6eff, 0x002e6fff, 0x002f6fff, 0x002f6fff,
        0x00306fff, 0x00316fff, 0x00316fff, 0x00326eff, 0x00336eff, 0x00346eff, 0x00346eff, 0x01356eff,
        0x06366eff, 0x0a376dff, 0x0e376dff, 0x12386dff, 0x15396dff, 0x17396dff, 0x1a3a6cff, 0x1c3b6cff,
        0x1e3c6cff, 0x203c6cff, 0x223d6cff, 0x243e6cff, 0x263e6cff, 0x273f6cff, 0x29406bff, 0x2b416bff,
        0x2c416bff, 0x2e426bff, 0x2f436bff, 0x31446bff, 0x32446bff, 0x33456bff, 0x35466bff, 0x36466bff,
        0x37476bff, 0x38486bff, 0x3a496bff, 0x3b496bff, 0x3c4a6bff, 0x3d4b6bff, 0x3e4b6bff, 0x404c6bff,
        0x414d6bff, 0x424e6bff, 0x434e6bff, 0x444f6bff, 0x45506bff, 0x46506bff, 0x47516bff, 0x48526bff,
        0x49536bff, 0x4a536bff, 0x4b546bff, 0x4c556bff, 0x4d556bff, 0x4e566bff, 0x4f576cff, 0x50586cff,
        0x51586cff, 0x52596cff, 0x535a6cff, 0x545a6cff, 0x555b6cff, 0x565c6cff, 0x575d6dff, 0x585d6dff,
        0x595e6dff, 0x5a5f6dff, 0x5b5f6dff, 0x5c606dff, 0x5d616eff, 0x5e626eff, 0x5f626eff, 0x5f636eff,
        0x60646eff, 0x61656fff, 0x62656fff, 0x63666fff, 0x64676fff, 0x65676fff, 0x666870ff, 0x676970ff,
        0x686a70ff, 0x686a70ff, 0x696b71ff, 0x6a6c71ff, 0x6b6d71ff, 0x6c6d72ff, 0x6d6e72ff, 0x6e6f72ff,
        0x6f6f72ff, 0x6f7073ff, 0x707173ff, 0x717273ff, 0x727274ff, 0x737374ff, 0x747475ff, 0x757575ff,
        0x757575ff, 0x767676ff, 0x777776ff, 0x787876ff, 0x797877ff, 0x7a7977ff, 0x7b7a77ff, 0x7b7b78ff,
        0x7c7b78ff, 0x7d7c78ff, 0x7e7d78ff, 0x7f7e78ff, 0x807e78ff, 0x817f78ff, 0x828078ff, 0x838178ff,
        0x848178ff, 0x858278ff, 0x868378ff, 0x878478ff, 0x888578ff, 0x898578ff, 0x8a8678ff, 0x8b8778ff,
        0x8c8878ff, 0x8d8878ff, 0x8e8978ff, 0x8f8a78ff, 0x908b78ff, 0x918c78ff, 0x928c78ff, 0x938d78ff,
        0x948e78ff, 0x958f78ff, 0x968f77ff, 0x979077ff, 0x989177ff, 0x999277ff, 0x9a9377ff, 0x9b9377ff,
        0x9c9477ff, 0x9d9577ff, 0x9e9676ff, 0x9f9776ff, 0xa09876ff, 0xa19876ff, 0xa29976ff, 0xa39a75ff,
        0xa49b75ff, 0xa59c75ff, 0xa69c75ff, 0xa79d75ff, 0xa89e74ff, 0xa99f74ff, 0xaaa074ff, 0xaba174ff,
        0xaca173ff, 0xada273ff, 0xaea373ff, 0xafa473ff, 0xb0a572ff, 0xb1a672ff, 0xb2a672ff, 0xb4a771ff,
        0xb5a871ff, 0xb6a971ff, 0xb7aa70ff, 0xb8ab70ff, 0xb9ab70ff, 0xbaac6fff, 0xbbad6fff, 0xbcae6eff,
        0xbdaf6eff, 0xbeb06eff, 0xbfb16dff, 0xc0b16dff, 0xc1b26cff, 0xc2b36cff, 0xc3b46cff, 0xc5b56bff,
        0xc6b66bff, 0xc7b76aff, 0xc8b86aff, 0xc9b869ff, 0xcab969ff, 0xcbba68ff, 0xccbb68ff, 0xcdbc67ff,
        0xcebd67ff, 0xd0be66ff, 0xd1bf66ff, 0xd2c065ff, 0xd3c065ff, 0xd4c164ff, 0xd5c263ff, 0xd6c363ff,
        0xd7c462ff, 0xd8c561ff, 0xd9c661ff, 0xdbc760ff, 0xdcc860ff, 0xddc95fff, 0xdeca5eff, 0xdfcb5dff,
        0xe0cb5dff, 0xe1cc5cff, 0xe3cd5bff, 0xe4ce5bff, 0xe5cf5aff, 0xe6d059ff, 0xe7d158ff, 0xe8d257ff,
        0xe9d356ff, 0xebd456ff, 0xecd555ff, 0xedd654ff, 0xeed753ff, 0xefd852ff, 0xf0d951ff, 0xf1da50ff,
        0xf3db4fff, 0xf4dc4eff, 0xf5dd4dff, 0xf6de4cff, 0xf7df4bff, 0xf9e049ff, 0xfae048ff, 0xfbe147ff,
        0xfce246ff, 0xfde345ff, 0xffe443ff, 0xffe542ff, 0xffe642ff, 0xffe743ff, 0xffe844ff, 0xffe945ff,
    ];
    exports.Turbo3 = [0x30123bff, 0xa1fc3dff, 0x7a0402];
    exports.Turbo4 = [0x30123bff, 0x1ae4b6ff, 0xf9ba38ff, 0x7a0402];
    exports.Turbo5 = [0x30123bff, 0x2ab9edff, 0xa1fc3dff, 0xfb8022ff, 0x7a0402];
    exports.Turbo6 = [0x30123bff, 0x3e9bfeff, 0x46f783ff, 0xe1dc37ff, 0xef5a11ff, 0x7a0402];
    exports.Turbo7 = [0x30123bff, 0x4584f9ff, 0x1ae4b6ff, 0xa1fc3dff, 0xf9ba38ff, 0xe5460aff, 0x7a0402];
    exports.Turbo8 = [0x30123bff, 0x4675edff, 0x1ccdd7ff, 0x61fc6cff, 0xcfea34ff, 0xfe9b2dff, 0xdb3a07ff, 0x7a0402];
    exports.Turbo9 = [0x30123bff, 0x4668e0ff, 0x2ab9edff, 0x2ff09aff, 0xa1fc3dff, 0xecd139ff, 0xfb8022ff, 0xd23005ff, 0x7a0402];
    exports.Turbo10 = [0x30123bff, 0x4560d6ff, 0x36a8f9ff, 0x1ae4b6ff, 0x71fd5fff, 0xc5ef33ff, 0xf9ba38ff, 0xf66b18ff, 0xcb2b03ff, 0x7a0402];
    exports.Turbo11 = [0x30123bff, 0x4458cbff, 0x3e9bfeff, 0x18d5ccff, 0x46f783ff, 0xa1fc3dff, 0xe1dc37ff, 0xfda631ff, 0xef5a11ff, 0xc52602ff, 0x7a0402];
    exports.Turbo256 = [
        0x30123bff, 0x311542ff, 0x32184aff, 0x341b51ff, 0x351e58ff, 0x36215fff, 0x372365ff, 0x38266cff,
        0x392972ff, 0x3a2c79ff, 0x3b2f7fff, 0x3c3285ff, 0x3c358bff, 0x3d3791ff, 0x3e3a96ff, 0x3f3d9cff,
        0x4040a1ff, 0x4043a6ff, 0x4145abff, 0x4148b0ff, 0x424bb5ff, 0x434ebaff, 0x4350beff, 0x4353c2ff,
        0x4456c7ff, 0x4458cbff, 0x455bceff, 0x455ed2ff, 0x4560d6ff, 0x4563d9ff, 0x4666ddff, 0x4668e0ff,
        0x466be3ff, 0x466de6ff, 0x4670e8ff, 0x4673ebff, 0x4675edff, 0x4678f0ff, 0x467af2ff, 0x467df4ff,
        0x467ff6ff, 0x4682f8ff, 0x4584f9ff, 0x4587fbff, 0x4589fcff, 0x448cfdff, 0x438efdff, 0x4291feff,
        0x4193feff, 0x4096feff, 0x3f98feff, 0x3e9bfeff, 0x3c9dfdff, 0x3ba0fcff, 0x39a2fcff, 0x38a5fbff,
        0x36a8f9ff, 0x34aaf8ff, 0x33acf6ff, 0x31aff5ff, 0x2fb1f3ff, 0x2db4f1ff, 0x2bb6efff, 0x2ab9edff,
        0x28bbebff, 0x26bde9ff, 0x25c0e6ff, 0x23c2e4ff, 0x21c4e1ff, 0x20c6dfff, 0x1ec9dcff, 0x1dcbdaff,
        0x1ccdd7ff, 0x1bcfd4ff, 0x1ad1d2ff, 0x19d3cfff, 0x18d5ccff, 0x18d7caff, 0x17d9c7ff, 0x17dac4ff,
        0x17dcc2ff, 0x17debfff, 0x18e0bdff, 0x18e1baff, 0x19e3b8ff, 0x1ae4b6ff, 0x1be5b4ff, 0x1de7b1ff,
        0x1ee8afff, 0x20e9acff, 0x22eba9ff, 0x24eca6ff, 0x27eda3ff, 0x29eea0ff, 0x2cef9dff, 0x2ff09aff,
        0x32f197ff, 0x35f394ff, 0x38f491ff, 0x3bf48dff, 0x3ff58aff, 0x42f687ff, 0x46f783ff, 0x4af880ff,
        0x4df97cff, 0x51f979ff, 0x55fa76ff, 0x59fb72ff, 0x5dfb6fff, 0x61fc6cff, 0x65fc68ff, 0x69fd65ff,
        0x6dfd62ff, 0x71fd5fff, 0x74fe5cff, 0x78fe59ff, 0x7cfe56ff, 0x80fe53ff, 0x84fe50ff, 0x87fe4dff,
        0x8bfe4bff, 0x8efe48ff, 0x92fe46ff, 0x95fe44ff, 0x98fe42ff, 0x9bfd40ff, 0x9efd3eff, 0xa1fc3dff,
        0xa4fc3bff, 0xa6fb3aff, 0xa9fb39ff, 0xacfa37ff, 0xaef937ff, 0xb1f836ff, 0xb3f835ff, 0xb6f735ff,
        0xb9f534ff, 0xbbf434ff, 0xbef334ff, 0xc0f233ff, 0xc3f133ff, 0xc5ef33ff, 0xc8ee33ff, 0xcaed33ff,
        0xcdeb34ff, 0xcfea34ff, 0xd1e834ff, 0xd4e735ff, 0xd6e535ff, 0xd8e335ff, 0xdae236ff, 0xdde036ff,
        0xdfde36ff, 0xe1dc37ff, 0xe3da37ff, 0xe5d838ff, 0xe7d738ff, 0xe8d538ff, 0xead339ff, 0xecd139ff,
        0xedcf39ff, 0xefcd39ff, 0xf0cb3aff, 0xf2c83aff, 0xf3c63aff, 0xf4c43aff, 0xf6c23aff, 0xf7c039ff,
        0xf8be39ff, 0xf9bc39ff, 0xf9ba38ff, 0xfab737ff, 0xfbb537ff, 0xfbb336ff, 0xfcb035ff, 0xfcae34ff,
        0xfdab33ff, 0xfda932ff, 0xfda631ff, 0xfda330ff, 0xfea12fff, 0xfe9e2eff, 0xfe9b2dff, 0xfe982cff,
        0xfd952bff, 0xfd9229ff, 0xfd8f28ff, 0xfd8c27ff, 0xfc8926ff, 0xfc8624ff, 0xfb8323ff, 0xfb8022ff,
        0xfa7d20ff, 0xfa7a1fff, 0xf9771eff, 0xf8741cff, 0xf7711bff, 0xf76e1aff, 0xf66b18ff, 0xf56817ff,
        0xf46516ff, 0xf36315ff, 0xf26014ff, 0xf15d13ff, 0xef5a11ff, 0xee5810ff, 0xed550fff, 0xec520eff,
        0xea500dff, 0xe94d0dff, 0xe84b0cff, 0xe6490bff, 0xe5460aff, 0xe3440aff, 0xe24209ff, 0xe04008ff,
        0xde3e08ff, 0xdd3c07ff, 0xdb3a07ff, 0xd93806ff, 0xd73606ff, 0xd63405ff, 0xd43205ff, 0xd23005ff,
        0xd02f04ff, 0xce2d04ff, 0xcb2b03ff, 0xc92903ff, 0xc72803ff, 0xc52602ff, 0xc32402ff, 0xc02302ff,
        0xbe2102ff, 0xbb1f01ff, 0xb91e01ff, 0xb61c01ff, 0xb41b01ff, 0xb11901ff, 0xae1801ff, 0xac1601ff,
        0xa91501ff, 0xa61401ff, 0xa31201ff, 0xa01101ff, 0x9d1001ff, 0x9a0e01ff, 0x970d01ff, 0x940c01ff,
        0x910b01ff, 0x8e0a01ff, 0x8b0901ff, 0x870801ff, 0x840701ff, 0x810602ff, 0x7d0502ff, 0x7a0402ff,
    ];
    exports.Accent3 = [0x7fc97fff, 0xbeaed4ff, 0xfdc086ff];
    exports.Accent4 = [0x7fc97fff, 0xbeaed4ff, 0xfdc086ff, 0xffff99ff];
    exports.Accent5 = [0x7fc97fff, 0xbeaed4ff, 0xfdc086ff, 0xffff99ff, 0x386cb0ff];
    exports.Accent6 = [0x7fc97fff, 0xbeaed4ff, 0xfdc086ff, 0xffff99ff, 0x386cb0ff, 0xf0027fff];
    exports.Accent7 = [0x7fc97fff, 0xbeaed4ff, 0xfdc086ff, 0xffff99ff, 0x386cb0ff, 0xf0027fff, 0xbf5b17ff];
    exports.Accent8 = [0x7fc97fff, 0xbeaed4ff, 0xfdc086ff, 0xffff99ff, 0x386cb0ff, 0xf0027fff, 0xbf5b17ff, 0x666666ff];
    exports.Dark2_3 = [0x1b9e77ff, 0xd95f02ff, 0x7570b3ff];
    exports.Dark2_4 = [0x1b9e77ff, 0xd95f02ff, 0x7570b3ff, 0xe7298aff];
    exports.Dark2_5 = [0x1b9e77ff, 0xd95f02ff, 0x7570b3ff, 0xe7298aff, 0x66a61eff];
    exports.Dark2_6 = [0x1b9e77ff, 0xd95f02ff, 0x7570b3ff, 0xe7298aff, 0x66a61eff, 0xe6ab02ff];
    exports.Dark2_7 = [0x1b9e77ff, 0xd95f02ff, 0x7570b3ff, 0xe7298aff, 0x66a61eff, 0xe6ab02ff, 0xa6761dff];
    exports.Dark2_8 = [0x1b9e77ff, 0xd95f02ff, 0x7570b3ff, 0xe7298aff, 0x66a61eff, 0xe6ab02ff, 0xa6761dff, 0x666666ff];
    exports.Paired3 = [0xa6cee3ff, 0x1f78b4ff, 0xb2df8aff];
    exports.Paired4 = [0xa6cee3ff, 0x1f78b4ff, 0xb2df8aff, 0x33a02cff];
    exports.Paired5 = [0xa6cee3ff, 0x1f78b4ff, 0xb2df8aff, 0x33a02cff, 0xfb9a99ff];
    exports.Paired6 = [0xa6cee3ff, 0x1f78b4ff, 0xb2df8aff, 0x33a02cff, 0xfb9a99ff, 0xe31a1cff];
    exports.Paired7 = [0xa6cee3ff, 0x1f78b4ff, 0xb2df8aff, 0x33a02cff, 0xfb9a99ff, 0xe31a1cff, 0xfdbf6fff];
    exports.Paired8 = [0xa6cee3ff, 0x1f78b4ff, 0xb2df8aff, 0x33a02cff, 0xfb9a99ff, 0xe31a1cff, 0xfdbf6fff, 0xff7f00ff];
    exports.Paired9 = [0xa6cee3ff, 0x1f78b4ff, 0xb2df8aff, 0x33a02cff, 0xfb9a99ff, 0xe31a1cff, 0xfdbf6fff, 0xff7f00ff, 0xcab2d6ff];
    exports.Paired10 = [0xa6cee3ff, 0x1f78b4ff, 0xb2df8aff, 0x33a02cff, 0xfb9a99ff, 0xe31a1cff, 0xfdbf6fff, 0xff7f00ff, 0xcab2d6ff, 0x6a3d9aff];
    exports.Paired11 = [0xa6cee3ff, 0x1f78b4ff, 0xb2df8aff, 0x33a02cff, 0xfb9a99ff, 0xe31a1cff, 0xfdbf6fff, 0xff7f00ff, 0xcab2d6ff, 0x6a3d9aff, 0xffff99ff];
    exports.Paired12 = [0xa6cee3ff, 0x1f78b4ff, 0xb2df8aff, 0x33a02cff, 0xfb9a99ff, 0xe31a1cff, 0xfdbf6fff, 0xff7f00ff, 0xcab2d6ff, 0x6a3d9aff, 0xffff99ff, 0xb15928ff];
    exports.Pastel1_3 = [0xfbb4aeff, 0xb3cde3ff, 0xccebc5ff];
    exports.Pastel1_4 = [0xfbb4aeff, 0xb3cde3ff, 0xccebc5ff, 0xdecbe4ff];
    exports.Pastel1_5 = [0xfbb4aeff, 0xb3cde3ff, 0xccebc5ff, 0xdecbe4ff, 0xfed9a6ff];
    exports.Pastel1_6 = [0xfbb4aeff, 0xb3cde3ff, 0xccebc5ff, 0xdecbe4ff, 0xfed9a6ff, 0xffffccff];
    exports.Pastel1_7 = [0xfbb4aeff, 0xb3cde3ff, 0xccebc5ff, 0xdecbe4ff, 0xfed9a6ff, 0xffffccff, 0xe5d8bdff];
    exports.Pastel1_8 = [0xfbb4aeff, 0xb3cde3ff, 0xccebc5ff, 0xdecbe4ff, 0xfed9a6ff, 0xffffccff, 0xe5d8bdff, 0xfddaecff];
    exports.Pastel1_9 = [0xfbb4aeff, 0xb3cde3ff, 0xccebc5ff, 0xdecbe4ff, 0xfed9a6ff, 0xffffccff, 0xe5d8bdff, 0xfddaecff, 0xf2f2f2ff];
    exports.Pastel2_3 = [0xb3e2cdff, 0xfdcdacff, 0xcbd5e8ff];
    exports.Pastel2_4 = [0xb3e2cdff, 0xfdcdacff, 0xcbd5e8ff, 0xf4cae4ff];
    exports.Pastel2_5 = [0xb3e2cdff, 0xfdcdacff, 0xcbd5e8ff, 0xf4cae4ff, 0xe6f5c9ff];
    exports.Pastel2_6 = [0xb3e2cdff, 0xfdcdacff, 0xcbd5e8ff, 0xf4cae4ff, 0xe6f5c9ff, 0xfff2aeff];
    exports.Pastel2_7 = [0xb3e2cdff, 0xfdcdacff, 0xcbd5e8ff, 0xf4cae4ff, 0xe6f5c9ff, 0xfff2aeff, 0xf1e2ccff];
    exports.Pastel2_8 = [0xb3e2cdff, 0xfdcdacff, 0xcbd5e8ff, 0xf4cae4ff, 0xe6f5c9ff, 0xfff2aeff, 0xf1e2ccff, 0xccccccff];
    exports.Set1_3 = [0xe41a1cff, 0x377eb8ff, 0x4daf4aff];
    exports.Set1_4 = [0xe41a1cff, 0x377eb8ff, 0x4daf4aff, 0x984ea3ff];
    exports.Set1_5 = [0xe41a1cff, 0x377eb8ff, 0x4daf4aff, 0x984ea3ff, 0xff7f00ff];
    exports.Set1_6 = [0xe41a1cff, 0x377eb8ff, 0x4daf4aff, 0x984ea3ff, 0xff7f00ff, 0xffff33ff];
    exports.Set1_7 = [0xe41a1cff, 0x377eb8ff, 0x4daf4aff, 0x984ea3ff, 0xff7f00ff, 0xffff33ff, 0xa65628ff];
    exports.Set1_8 = [0xe41a1cff, 0x377eb8ff, 0x4daf4aff, 0x984ea3ff, 0xff7f00ff, 0xffff33ff, 0xa65628ff, 0xf781bfff];
    exports.Set1_9 = [0xe41a1cff, 0x377eb8ff, 0x4daf4aff, 0x984ea3ff, 0xff7f00ff, 0xffff33ff, 0xa65628ff, 0xf781bfff, 0x999999ff];
    exports.Set2_3 = [0x66c2a5ff, 0xfc8d62ff, 0x8da0cbff];
    exports.Set2_4 = [0x66c2a5ff, 0xfc8d62ff, 0x8da0cbff, 0xe78ac3ff];
    exports.Set2_5 = [0x66c2a5ff, 0xfc8d62ff, 0x8da0cbff, 0xe78ac3ff, 0xa6d854ff];
    exports.Set2_6 = [0x66c2a5ff, 0xfc8d62ff, 0x8da0cbff, 0xe78ac3ff, 0xa6d854ff, 0xffd92fff];
    exports.Set2_7 = [0x66c2a5ff, 0xfc8d62ff, 0x8da0cbff, 0xe78ac3ff, 0xa6d854ff, 0xffd92fff, 0xe5c494ff];
    exports.Set2_8 = [0x66c2a5ff, 0xfc8d62ff, 0x8da0cbff, 0xe78ac3ff, 0xa6d854ff, 0xffd92fff, 0xe5c494ff, 0xb3b3b3ff];
    exports.Set3_3 = [0x8dd3c7ff, 0xffffb3ff, 0xbebadaff];
    exports.Set3_4 = [0x8dd3c7ff, 0xffffb3ff, 0xbebadaff, 0xfb8072ff];
    exports.Set3_5 = [0x8dd3c7ff, 0xffffb3ff, 0xbebadaff, 0xfb8072ff, 0x80b1d3ff];
    exports.Set3_6 = [0x8dd3c7ff, 0xffffb3ff, 0xbebadaff, 0xfb8072ff, 0x80b1d3ff, 0xfdb462ff];
    exports.Set3_7 = [0x8dd3c7ff, 0xffffb3ff, 0xbebadaff, 0xfb8072ff, 0x80b1d3ff, 0xfdb462ff, 0xb3de69ff];
    exports.Set3_8 = [0x8dd3c7ff, 0xffffb3ff, 0xbebadaff, 0xfb8072ff, 0x80b1d3ff, 0xfdb462ff, 0xb3de69ff, 0xfccde5ff];
    exports.Set3_9 = [0x8dd3c7ff, 0xffffb3ff, 0xbebadaff, 0xfb8072ff, 0x80b1d3ff, 0xfdb462ff, 0xb3de69ff, 0xfccde5ff, 0xd9d9d9ff];
    exports.Set3_10 = [0x8dd3c7ff, 0xffffb3ff, 0xbebadaff, 0xfb8072ff, 0x80b1d3ff, 0xfdb462ff, 0xb3de69ff, 0xfccde5ff, 0xd9d9d9ff, 0xbc80bdff];
    exports.Set3_11 = [0x8dd3c7ff, 0xffffb3ff, 0xbebadaff, 0xfb8072ff, 0x80b1d3ff, 0xfdb462ff, 0xb3de69ff, 0xfccde5ff, 0xd9d9d9ff, 0xbc80bdff, 0xccebc5ff];
    exports.Set3_12 = [0x8dd3c7ff, 0xffffb3ff, 0xbebadaff, 0xfb8072ff, 0x80b1d3ff, 0xfdb462ff, 0xb3de69ff, 0xfccde5ff, 0xd9d9d9ff, 0xbc80bdff, 0xccebc5ff, 0xffed6fff];
    exports.Category10_3 = [0x1f77b4ff, 0xff7f0eff, 0x2ca02cff];
    exports.Category10_4 = [0x1f77b4ff, 0xff7f0eff, 0x2ca02cff, 0xd62728ff];
    exports.Category10_5 = [0x1f77b4ff, 0xff7f0eff, 0x2ca02cff, 0xd62728ff, 0x9467bdff];
    exports.Category10_6 = [0x1f77b4ff, 0xff7f0eff, 0x2ca02cff, 0xd62728ff, 0x9467bdff, 0x8c564bff];
    exports.Category10_7 = [0x1f77b4ff, 0xff7f0eff, 0x2ca02cff, 0xd62728ff, 0x9467bdff, 0x8c564bff, 0xe377c2ff];
    exports.Category10_8 = [0x1f77b4ff, 0xff7f0eff, 0x2ca02cff, 0xd62728ff, 0x9467bdff, 0x8c564bff, 0xe377c2ff, 0x7f7f7fff];
    exports.Category10_9 = [0x1f77b4ff, 0xff7f0eff, 0x2ca02cff, 0xd62728ff, 0x9467bdff, 0x8c564bff, 0xe377c2ff, 0x7f7f7fff, 0xbcbd22ff];
    exports.Category10_10 = [0x1f77b4ff, 0xff7f0eff, 0x2ca02cff, 0xd62728ff, 0x9467bdff, 0x8c564bff, 0xe377c2ff, 0x7f7f7fff, 0xbcbd22ff, 0x17becfff];
    exports.Category20_3 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff];
    exports.Category20_4 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff];
    exports.Category20_5 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff];
    exports.Category20_6 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff];
    exports.Category20_7 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff];
    exports.Category20_8 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff];
    exports.Category20_9 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff];
    exports.Category20_10 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff];
    exports.Category20_11 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff,
        0x8c564bff];
    exports.Category20_12 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff,
        0x8c564bff, 0xc49c94ff];
    exports.Category20_13 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff,
        0x8c564bff, 0xc49c94ff, 0xe377c2ff];
    exports.Category20_14 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff,
        0x8c564bff, 0xc49c94ff, 0xe377c2ff, 0xf7b6d2ff];
    exports.Category20_15 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff,
        0x8c564bff, 0xc49c94ff, 0xe377c2ff, 0xf7b6d2ff, 0x7f7f7fff];
    exports.Category20_16 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff,
        0x8c564bff, 0xc49c94ff, 0xe377c2ff, 0xf7b6d2ff, 0x7f7f7fff, 0xc7c7c7ff];
    exports.Category20_17 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff,
        0x8c564bff, 0xc49c94ff, 0xe377c2ff, 0xf7b6d2ff, 0x7f7f7fff, 0xc7c7c7ff, 0xbcbd22ff];
    exports.Category20_18 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff,
        0x8c564bff, 0xc49c94ff, 0xe377c2ff, 0xf7b6d2ff, 0x7f7f7fff, 0xc7c7c7ff, 0xbcbd22ff, 0xdbdb8dff];
    exports.Category20_19 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff,
        0x8c564bff, 0xc49c94ff, 0xe377c2ff, 0xf7b6d2ff, 0x7f7f7fff, 0xc7c7c7ff, 0xbcbd22ff, 0xdbdb8dff, 0x17becfff];
    exports.Category20_20 = [0x1f77b4ff, 0xaec7e8ff, 0xff7f0eff, 0xffbb78ff, 0x2ca02cff, 0x98df8aff, 0xd62728ff, 0xff9896ff, 0x9467bdff, 0xc5b0d5ff,
        0x8c564bff, 0xc49c94ff, 0xe377c2ff, 0xf7b6d2ff, 0x7f7f7fff, 0xc7c7c7ff, 0xbcbd22ff, 0xdbdb8dff, 0x17becfff, 0x9edae5ff];
    exports.Category20b_3 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff];
    exports.Category20b_4 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff];
    exports.Category20b_5 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff];
    exports.Category20b_6 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff];
    exports.Category20b_7 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff];
    exports.Category20b_8 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff];
    exports.Category20b_9 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff];
    exports.Category20b_10 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff];
    exports.Category20b_11 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff,
        0xe7ba52ff];
    exports.Category20b_12 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff,
        0xe7ba52ff, 0xe7cb94ff];
    exports.Category20b_13 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff,
        0xe7ba52ff, 0xe7cb94ff, 0x843c39ff];
    exports.Category20b_14 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff,
        0xe7ba52ff, 0xe7cb94ff, 0x843c39ff, 0xad494aff];
    exports.Category20b_15 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff,
        0xe7ba52ff, 0xe7cb94ff, 0x843c39ff, 0xad494aff, 0xd6616bff];
    exports.Category20b_16 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff,
        0xe7ba52ff, 0xe7cb94ff, 0x843c39ff, 0xad494aff, 0xd6616bff, 0xe7969cff];
    exports.Category20b_17 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff,
        0xe7ba52ff, 0xe7cb94ff, 0x843c39ff, 0xad494aff, 0xd6616bff, 0xe7969cff, 0x7b4173ff];
    exports.Category20b_18 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff,
        0xe7ba52ff, 0xe7cb94ff, 0x843c39ff, 0xad494aff, 0xd6616bff, 0xe7969cff, 0x7b4173ff, 0xa55194ff];
    exports.Category20b_19 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff,
        0xe7ba52ff, 0xe7cb94ff, 0x843c39ff, 0xad494aff, 0xd6616bff, 0xe7969cff, 0x7b4173ff, 0xa55194ff, 0xce6dbdff];
    exports.Category20b_20 = [0x393b79ff, 0x5254a3ff, 0x6b6ecfff, 0x9c9edeff, 0x637939ff, 0x8ca252ff, 0xb5cf6bff, 0xcedb9cff, 0x8c6d31ff, 0xbd9e39ff,
        0xe7ba52ff, 0xe7cb94ff, 0x843c39ff, 0xad494aff, 0xd6616bff, 0xe7969cff, 0x7b4173ff, 0xa55194ff, 0xce6dbdff, 0xde9ed6ff];
    exports.Category20c_3 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff];
    exports.Category20c_4 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff];
    exports.Category20c_5 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff];
    exports.Category20c_6 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff];
    exports.Category20c_7 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff];
    exports.Category20c_8 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff];
    exports.Category20c_9 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff];
    exports.Category20c_10 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff];
    exports.Category20c_11 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff,
        0xa1d99bff];
    exports.Category20c_12 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff,
        0xa1d99bff, 0xc7e9c0ff];
    exports.Category20c_13 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff,
        0xa1d99bff, 0xc7e9c0ff, 0x756bb1ff];
    exports.Category20c_14 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff,
        0xa1d99bff, 0xc7e9c0ff, 0x756bb1ff, 0x9e9ac8ff];
    exports.Category20c_15 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff,
        0xa1d99bff, 0xc7e9c0ff, 0x756bb1ff, 0x9e9ac8ff, 0xbcbddcff];
    exports.Category20c_16 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff,
        0xa1d99bff, 0xc7e9c0ff, 0x756bb1ff, 0x9e9ac8ff, 0xbcbddcff, 0xdadaebff];
    exports.Category20c_17 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff,
        0xa1d99bff, 0xc7e9c0ff, 0x756bb1ff, 0x9e9ac8ff, 0xbcbddcff, 0xdadaebff, 0x636363ff];
    exports.Category20c_18 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff,
        0xa1d99bff, 0xc7e9c0ff, 0x756bb1ff, 0x9e9ac8ff, 0xbcbddcff, 0xdadaebff, 0x636363ff, 0x969696ff];
    exports.Category20c_19 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff,
        0xa1d99bff, 0xc7e9c0ff, 0x756bb1ff, 0x9e9ac8ff, 0xbcbddcff, 0xdadaebff, 0x636363ff, 0x969696ff, 0xbdbdbdff];
    exports.Category20c_20 = [0x3182bdff, 0x6baed6ff, 0x9ecae1ff, 0xc6dbefff, 0xe6550dff, 0xfd8d3cff, 0xfdae6bff, 0xfdd0a2ff, 0x31a354ff, 0x74c476ff,
        0xa1d99bff, 0xc7e9c0ff, 0x756bb1ff, 0x9e9ac8ff, 0xbcbddcff, 0xdadaebff, 0x636363ff, 0x969696ff, 0xbdbdbdff, 0xd9d9d9ff];
    exports.Colorblind3 = [0x0072b2ff, 0xe69f00ff, 0xf0e442ff];
    exports.Colorblind4 = [0x0072b2ff, 0xe69f00ff, 0xf0e442ff, 0x009e73ff];
    exports.Colorblind5 = [0x0072b2ff, 0xe69f00ff, 0xf0e442ff, 0x009e73ff, 0x56b4e9ff];
    exports.Colorblind6 = [0x0072b2ff, 0xe69f00ff, 0xf0e442ff, 0x009e73ff, 0x56b4e9ff, 0xd55e00ff];
    exports.Colorblind7 = [0x0072b2ff, 0xe69f00ff, 0xf0e442ff, 0x009e73ff, 0x56b4e9ff, 0xd55e00ff, 0xcc79a7ff];
    exports.Colorblind8 = [0x0072b2ff, 0xe69f00ff, 0xf0e442ff, 0x009e73ff, 0x56b4e9ff, 0xd55e00ff, 0xcc79a7ff, 0x000000ff];
    exports.YlGn = { YlGn3: exports.YlGn3, YlGn4: exports.YlGn4, YlGn5: exports.YlGn5, YlGn6: exports.YlGn6, YlGn7: exports.YlGn7, YlGn8: exports.YlGn8, YlGn9: exports.YlGn9 };
    exports.YlGnBu = { YlGnBu3: exports.YlGnBu3, YlGnBu4: exports.YlGnBu4, YlGnBu5: exports.YlGnBu5, YlGnBu6: exports.YlGnBu6, YlGnBu7: exports.YlGnBu7, YlGnBu8: exports.YlGnBu8, YlGnBu9: exports.YlGnBu9 };
    exports.GnBu = { GnBu3: exports.GnBu3, GnBu4: exports.GnBu4, GnBu5: exports.GnBu5, GnBu6: exports.GnBu6, GnBu7: exports.GnBu7, GnBu8: exports.GnBu8, GnBu9: exports.GnBu9 };
    exports.BuGn = { BuGn3: exports.BuGn3, BuGn4: exports.BuGn4, BuGn5: exports.BuGn5, BuGn6: exports.BuGn6, BuGn7: exports.BuGn7, BuGn8: exports.BuGn8, BuGn9: exports.BuGn9 };
    exports.PuBuGn = { PuBuGn3: exports.PuBuGn3, PuBuGn4: exports.PuBuGn4, PuBuGn5: exports.PuBuGn5, PuBuGn6: exports.PuBuGn6, PuBuGn7: exports.PuBuGn7, PuBuGn8: exports.PuBuGn8, PuBuGn9: exports.PuBuGn9 };
    exports.PuBu = { PuBu3: exports.PuBu3, PuBu4: exports.PuBu4, PuBu5: exports.PuBu5, PuBu6: exports.PuBu6, PuBu7: exports.PuBu7, PuBu8: exports.PuBu8, PuBu9: exports.PuBu9 };
    exports.BuPu = { BuPu3: exports.BuPu3, BuPu4: exports.BuPu4, BuPu5: exports.BuPu5, BuPu6: exports.BuPu6, BuPu7: exports.BuPu7, BuPu8: exports.BuPu8, BuPu9: exports.BuPu9 };
    exports.RdPu = { RdPu3: exports.RdPu3, RdPu4: exports.RdPu4, RdPu5: exports.RdPu5, RdPu6: exports.RdPu6, RdPu7: exports.RdPu7, RdPu8: exports.RdPu8, RdPu9: exports.RdPu9 };
    exports.PuRd = { PuRd3: exports.PuRd3, PuRd4: exports.PuRd4, PuRd5: exports.PuRd5, PuRd6: exports.PuRd6, PuRd7: exports.PuRd7, PuRd8: exports.PuRd8, PuRd9: exports.PuRd9 };
    exports.OrRd = { OrRd3: exports.OrRd3, OrRd4: exports.OrRd4, OrRd5: exports.OrRd5, OrRd6: exports.OrRd6, OrRd7: exports.OrRd7, OrRd8: exports.OrRd8, OrRd9: exports.OrRd9 };
    exports.YlOrRd = { YlOrRd3: exports.YlOrRd3, YlOrRd4: exports.YlOrRd4, YlOrRd5: exports.YlOrRd5, YlOrRd6: exports.YlOrRd6, YlOrRd7: exports.YlOrRd7, YlOrRd8: exports.YlOrRd8, YlOrRd9: exports.YlOrRd9 };
    exports.YlOrBr = { YlOrBr3: exports.YlOrBr3, YlOrBr4: exports.YlOrBr4, YlOrBr5: exports.YlOrBr5, YlOrBr6: exports.YlOrBr6, YlOrBr7: exports.YlOrBr7, YlOrBr8: exports.YlOrBr8, YlOrBr9: exports.YlOrBr9 };
    exports.Purples = { Purples3: exports.Purples3, Purples4: exports.Purples4, Purples5: exports.Purples5, Purples6: exports.Purples6, Purples7: exports.Purples7, Purples8: exports.Purples8, Purples9: exports.Purples9 };
    exports.Blues = { Blues3: exports.Blues3, Blues4: exports.Blues4, Blues5: exports.Blues5, Blues6: exports.Blues6, Blues7: exports.Blues7, Blues8: exports.Blues8, Blues9: exports.Blues9 };
    exports.Greens = { Greens3: exports.Greens3, Greens4: exports.Greens4, Greens5: exports.Greens5, Greens6: exports.Greens6, Greens7: exports.Greens7, Greens8: exports.Greens8, Greens9: exports.Greens9 };
    exports.Oranges = { Oranges3: exports.Oranges3, Oranges4: exports.Oranges4, Oranges5: exports.Oranges5, Oranges6: exports.Oranges6, Oranges7: exports.Oranges7, Oranges8: exports.Oranges8, Oranges9: exports.Oranges9 };
    exports.Reds = { Reds3: exports.Reds3, Reds4: exports.Reds4, Reds5: exports.Reds5, Reds6: exports.Reds6, Reds7: exports.Reds7, Reds8: exports.Reds8, Reds9: exports.Reds9 };
    exports.Greys = { Greys3: exports.Greys3, Greys4: exports.Greys4, Greys5: exports.Greys5, Greys6: exports.Greys6, Greys7: exports.Greys7, Greys8: exports.Greys8, Greys9: exports.Greys9, Greys10: exports.Greys10, Greys11: exports.Greys11, Greys256: exports.Greys256 };
    exports.PuOr = { PuOr3: exports.PuOr3, PuOr4: exports.PuOr4, PuOr5: exports.PuOr5, PuOr6: exports.PuOr6, PuOr7: exports.PuOr7, PuOr8: exports.PuOr8, PuOr9: exports.PuOr9, PuOr10: exports.PuOr10, PuOr11: exports.PuOr11 };
    exports.BrBG = { BrBG3: exports.BrBG3, BrBG4: exports.BrBG4, BrBG5: exports.BrBG5, BrBG6: exports.BrBG6, BrBG7: exports.BrBG7, BrBG8: exports.BrBG8, BrBG9: exports.BrBG9, BrBG10: exports.BrBG10, BrBG11: exports.BrBG11 };
    exports.PRGn = { PRGn3: exports.PRGn3, PRGn4: exports.PRGn4, PRGn5: exports.PRGn5, PRGn6: exports.PRGn6, PRGn7: exports.PRGn7, PRGn8: exports.PRGn8, PRGn9: exports.PRGn9, PRGn10: exports.PRGn10, PRGn11: exports.PRGn11 };
    exports.PiYG = { PiYG3: exports.PiYG3, PiYG4: exports.PiYG4, PiYG5: exports.PiYG5, PiYG6: exports.PiYG6, PiYG7: exports.PiYG7, PiYG8: exports.PiYG8, PiYG9: exports.PiYG9, PiYG10: exports.PiYG10, PiYG11: exports.PiYG11 };
    exports.RdBu = { RdBu3: exports.RdBu3, RdBu4: exports.RdBu4, RdBu5: exports.RdBu5, RdBu6: exports.RdBu6, RdBu7: exports.RdBu7, RdBu8: exports.RdBu8, RdBu9: exports.RdBu9, RdBu10: exports.RdBu10, RdBu11: exports.RdBu11 };
    exports.RdGy = { RdGy3: exports.RdGy3, RdGy4: exports.RdGy4, RdGy5: exports.RdGy5, RdGy6: exports.RdGy6, RdGy7: exports.RdGy7, RdGy8: exports.RdGy8, RdGy9: exports.RdGy9, RdGy10: exports.RdGy10, RdGy11: exports.RdGy11 };
    exports.RdYlBu = { RdYlBu3: exports.RdYlBu3, RdYlBu4: exports.RdYlBu4, RdYlBu5: exports.RdYlBu5, RdYlBu6: exports.RdYlBu6, RdYlBu7: exports.RdYlBu7, RdYlBu8: exports.RdYlBu8, RdYlBu9: exports.RdYlBu9, RdYlBu10: exports.RdYlBu10, RdYlBu11: exports.RdYlBu11 };
    exports.Spectral = { Spectral3: exports.Spectral3, Spectral4: exports.Spectral4, Spectral5: exports.Spectral5, Spectral6: exports.Spectral6, Spectral7: exports.Spectral7, Spectral8: exports.Spectral8, Spectral9: exports.Spectral9, Spectral10: exports.Spectral10, Spectral11: exports.Spectral11 };
    exports.RdYlGn = { RdYlGn3: exports.RdYlGn3, RdYlGn4: exports.RdYlGn4, RdYlGn5: exports.RdYlGn5, RdYlGn6: exports.RdYlGn6, RdYlGn7: exports.RdYlGn7, RdYlGn8: exports.RdYlGn8, RdYlGn9: exports.RdYlGn9, RdYlGn10: exports.RdYlGn10, RdYlGn11: exports.RdYlGn11 };
    exports.Inferno = { Inferno3: exports.Inferno3, Inferno4: exports.Inferno4, Inferno5: exports.Inferno5, Inferno6: exports.Inferno6, Inferno7: exports.Inferno7, Inferno8: exports.Inferno8, Inferno9: exports.Inferno9, Inferno10: exports.Inferno10, Inferno11: exports.Inferno11, Inferno256: exports.Inferno256 };
    exports.Magma = { Magma3: exports.Magma3, Magma4: exports.Magma4, Magma5: exports.Magma5, Magma6: exports.Magma6, Magma7: exports.Magma7, Magma8: exports.Magma8, Magma9: exports.Magma9, Magma10: exports.Magma10, Magma11: exports.Magma11, Magma256: exports.Magma256 };
    exports.Plasma = { Plasma3: exports.Plasma3, Plasma4: exports.Plasma4, Plasma5: exports.Plasma5, Plasma6: exports.Plasma6, Plasma7: exports.Plasma7, Plasma8: exports.Plasma8, Plasma9: exports.Plasma9, Plasma10: exports.Plasma10, Plasma11: exports.Plasma11, Plasma256: exports.Plasma256 };
    exports.Viridis = { Viridis3: exports.Viridis3, Viridis4: exports.Viridis4, Viridis5: exports.Viridis5, Viridis6: exports.Viridis6, Viridis7: exports.Viridis7, Viridis8: exports.Viridis8, Viridis9: exports.Viridis9, Viridis10: exports.Viridis10, Viridis11: exports.Viridis11, Viridis256: exports.Viridis256 };
    exports.Accent = { Accent3: exports.Accent3, Accent4: exports.Accent4, Accent5: exports.Accent5, Accent6: exports.Accent6, Accent7: exports.Accent7, Accent8: exports.Accent8 };
    exports.Dark2 = { Dark2_3: exports.Dark2_3, Dark2_4: exports.Dark2_4, Dark2_5: exports.Dark2_5, Dark2_6: exports.Dark2_6, Dark2_7: exports.Dark2_7, Dark2_8: exports.Dark2_8 };
    exports.Paired = { Paired3: exports.Paired3, Paired4: exports.Paired4, Paired5: exports.Paired5, Paired6: exports.Paired6, Paired7: exports.Paired7, Paired8: exports.Paired8, Paired9: exports.Paired9, Paired10: exports.Paired10, Paired11: exports.Paired11, Paired12: exports.Paired12 };
    exports.Pastel1 = { Pastel1_3: exports.Pastel1_3, Pastel1_4: exports.Pastel1_4, Pastel1_5: exports.Pastel1_5, Pastel1_6: exports.Pastel1_6, Pastel1_7: exports.Pastel1_7, Pastel1_8: exports.Pastel1_8, Pastel1_9: exports.Pastel1_9 };
    exports.Pastel2 = { Pastel2_3: exports.Pastel2_3, Pastel2_4: exports.Pastel2_4, Pastel2_5: exports.Pastel2_5, Pastel2_6: exports.Pastel2_6, Pastel2_7: exports.Pastel2_7, Pastel2_8: exports.Pastel2_8 };
    exports.Set1 = { Set1_3: exports.Set1_3, Set1_4: exports.Set1_4, Set1_5: exports.Set1_5, Set1_6: exports.Set1_6, Set1_7: exports.Set1_7, Set1_8: exports.Set1_8, Set1_9: exports.Set1_9 };
    exports.Set2 = { Set2_3: exports.Set2_3, Set2_4: exports.Set2_4, Set2_5: exports.Set2_5, Set2_6: exports.Set2_6, Set2_7: exports.Set2_7, Set2_8: exports.Set2_8 };
    exports.Set3 = { Set3_3: exports.Set3_3, Set3_4: exports.Set3_4, Set3_5: exports.Set3_5, Set3_6: exports.Set3_6, Set3_7: exports.Set3_7, Set3_8: exports.Set3_8, Set3_9: exports.Set3_9, Set3_10: exports.Set3_10, Set3_11: exports.Set3_11, Set3_12: exports.Set3_12 };
    exports.Category10 = { Category10_3: exports.Category10_3, Category10_4: exports.Category10_4, Category10_5: exports.Category10_5, Category10_6: exports.Category10_6, Category10_7: exports.Category10_7, Category10_8: exports.Category10_8, Category10_9: exports.Category10_9, Category10_10: exports.Category10_10 };
    exports.Category20 = { Category20_3: exports.Category20_3, Category20_4: exports.Category20_4, Category20_5: exports.Category20_5, Category20_6: exports.Category20_6, Category20_7: exports.Category20_7, Category20_8: exports.Category20_8, Category20_9: exports.Category20_9, Category20_10: exports.Category20_10, Category20_11: exports.Category20_11, Category20_12: exports.Category20_12, Category20_13: exports.Category20_13, Category20_14: exports.Category20_14, Category20_15: exports.Category20_15, Category20_16: exports.Category20_16, Category20_17: exports.Category20_17, Category20_18: exports.Category20_18, Category20_19: exports.Category20_19, Category20_20: exports.Category20_20 };
    exports.Category20b = { Category20b_3: exports.Category20b_3, Category20b_4: exports.Category20b_4, Category20b_5: exports.Category20b_5, Category20b_6: exports.Category20b_6, Category20b_7: exports.Category20b_7, Category20b_8: exports.Category20b_8, Category20b_9: exports.Category20b_9, Category20b_10: exports.Category20b_10, Category20b_11: exports.Category20b_11, Category20b_12: exports.Category20b_12, Category20b_13: exports.Category20b_13, Category20b_14: exports.Category20b_14, Category20b_15: exports.Category20b_15, Category20b_16: exports.Category20b_16, Category20b_17: exports.Category20b_17, Category20b_18: exports.Category20b_18, Category20b_19: exports.Category20b_19, Category20b_20: exports.Category20b_20 };
    exports.Category20c = { Category20c_3: exports.Category20c_3, Category20c_4: exports.Category20c_4, Category20c_5: exports.Category20c_5, Category20c_6: exports.Category20c_6, Category20c_7: exports.Category20c_7, Category20c_8: exports.Category20c_8, Category20c_9: exports.Category20c_9, Category20c_10: exports.Category20c_10, Category20c_11: exports.Category20c_11, Category20c_12: exports.Category20c_12, Category20c_13: exports.Category20c_13, Category20c_14: exports.Category20c_14, Category20c_15: exports.Category20c_15, Category20c_16: exports.Category20c_16, Category20c_17: exports.Category20c_17, Category20c_18: exports.Category20c_18, Category20c_19: exports.Category20c_19, Category20c_20: exports.Category20c_20 };
    exports.Colorblind = { Colorblind3: exports.Colorblind3, Colorblind4: exports.Colorblind4, Colorblind5: exports.Colorblind5, Colorblind6: exports.Colorblind6, Colorblind7: exports.Colorblind7, Colorblind8: exports.Colorblind8 };
    function linear_palette(palette, n) {
        if (n <= palette.length)
            return (0, array_1.linspace)(0, palette.length - 1, n).map((i) => palette[i | 0]);
        else
            throw new Error("too many color entries requested");
    }
    exports.linear_palette = linear_palette;
    function magma(n) {
        return linear_palette(exports.Magma256, n);
    }
    exports.magma = magma;
    function inferno(n) {
        return linear_palette(exports.Inferno256, n);
    }
    exports.inferno = inferno;
    function plasma(n) {
        return linear_palette(exports.Plasma256, n);
    }
    exports.plasma = plasma;
    function viridis(n) {
        return linear_palette(exports.Viridis256, n);
    }
    exports.viridis = viridis;
    function cividis(n) {
        return linear_palette(exports.Cividis256, n);
    }
    exports.cividis = cividis;
    function turbo(n) {
        return linear_palette(exports.Turbo256, n);
    }
    exports.turbo = turbo;
    function grey(n) {
        return linear_palette(exports.Greys256, n);
    }
    exports.grey = grey;
},
432: /* api/models.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    const tslib_1 = require(1) /* tslib */;
    (0, tslib_1.__exportStar)(require(38) /* ../models */, exports);
},
433: /* api/plotting.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    const tslib_1 = require(1) /* tslib */;
    const document_1 = require(5) /* ../document */;
    const embed = (0, tslib_1.__importStar)(require(4) /* ../embed */);
    const logging_1 = require(19) /* ../core/logging */;
    const properties_1 = require(18) /* ../core/properties */;
    const eq_1 = require(26) /* ../core/util/eq */;
    const array_1 = require(9) /* ../core/util/array */;
    const object_1 = require(13) /* ../core/util/object */;
    const types_1 = require(8) /* ../core/util/types */;
    const dom_1 = require(43) /* ../core/dom */;
    const iterator_1 = require(234) /* ../core/util/iterator */;
    const nd = (0, tslib_1.__importStar)(require(29) /* ../core/util/ndarray */);
    const models_1 = require(432) /* ./models */;
    const glyphs_1 = require(267) /* ../models/glyphs */;
    const legend_1 = require(214) /* ../models/annotations/legend */;
    const legend_item_1 = require(215) /* ../models/annotations/legend_item */;
    var gridplot_1 = require(434) /* ./gridplot */;
    __esExport("gridplot", gridplot_1.gridplot);
    var color_1 = require(22) /* ../core/util/color */;
    __esExport("color", color_1.color2css);
    const { hasOwnProperty } = Object.prototype;
    const _default_tools = ["pan", "wheel_zoom", "box_zoom", "save", "reset", "help"];
    const _default_color = "#1f77b4";
    const _default_alpha = 1.0;
    function _with_default(value, default_value) {
        return value === undefined ? default_value : value;
    }
    class Figure extends models_1.Plot {
        constructor(attrs = {}) {
            attrs = Object.assign({}, attrs);
            const tools = _with_default(attrs.tools, _default_tools);
            delete attrs.tools;
            const x_axis_type = _with_default(attrs.x_axis_type, "auto");
            const y_axis_type = _with_default(attrs.y_axis_type, "auto");
            delete attrs.x_axis_type;
            delete attrs.y_axis_type;
            const x_minor_ticks = attrs.x_minor_ticks != null ? attrs.x_minor_ticks : "auto";
            const y_minor_ticks = attrs.y_minor_ticks != null ? attrs.y_minor_ticks : "auto";
            delete attrs.x_minor_ticks;
            delete attrs.y_minor_ticks;
            const x_axis_location = attrs.x_axis_location != null ? attrs.x_axis_location : "below";
            const y_axis_location = attrs.y_axis_location != null ? attrs.y_axis_location : "left";
            delete attrs.x_axis_location;
            delete attrs.y_axis_location;
            const x_axis_label = attrs.x_axis_label != null ? attrs.x_axis_label : "";
            const y_axis_label = attrs.y_axis_label != null ? attrs.y_axis_label : "";
            delete attrs.x_axis_label;
            delete attrs.y_axis_label;
            const x_range = Figure._get_range(attrs.x_range);
            const y_range = Figure._get_range(attrs.y_range);
            delete attrs.x_range;
            delete attrs.y_range;
            const x_scale = attrs.x_scale != null ? attrs.x_scale : Figure._get_scale(x_range, x_axis_type);
            const y_scale = attrs.y_scale != null ? attrs.y_scale : Figure._get_scale(y_range, y_axis_type);
            delete attrs.x_scale;
            delete attrs.y_scale;
            super(Object.assign(Object.assign({}, attrs), { x_range, y_range, x_scale, y_scale }));
            this._process_axis_and_grid(x_axis_type, x_axis_location, x_minor_ticks, x_axis_label, x_range, 0);
            this._process_axis_and_grid(y_axis_type, y_axis_location, y_minor_ticks, y_axis_label, y_range, 1);
            this.add_tools(...this._process_tools(tools));
        }
        get xgrid() {
            return this.center.filter((r) => r instanceof models_1.Grid && r.dimension == 0);
        }
        get ygrid() {
            return this.center.filter((r) => r instanceof models_1.Grid && r.dimension == 1);
        }
        get xaxis() {
            return [...this.below, ...this.above].filter((r) => r instanceof models_1.Axis);
        }
        get yaxis() {
            return [...this.left, ...this.right].filter((r) => r instanceof models_1.Axis);
        }
        get grid() {
            return this.center.filter((r) => r instanceof models_1.Grid);
        }
        get axis() {
            return [...this.below, ...this.above, ...this.left, ...this.right].filter((r) => r instanceof models_1.Axis);
        }
        get legend() {
            const legends = this.panels.filter((r) => r instanceof legend_1.Legend);
            if (legends.length == 0) {
                const legend = new legend_1.Legend();
                this.add_layout(legend);
                return legend;
            }
            else {
                const [legend] = legends;
                return legend;
            }
        }
        annular_wedge(...args) {
            return this._glyph(glyphs_1.AnnularWedge, "x,y,inner_radius,outer_radius,start_angle,end_angle", args);
        }
        annulus(...args) {
            return this._glyph(glyphs_1.Annulus, "x,y,inner_radius,outer_radius", args);
        }
        arc(...args) {
            return this._glyph(glyphs_1.Arc, "x,y,radius,start_angle,end_angle", args);
        }
        bezier(...args) {
            return this._glyph(glyphs_1.Bezier, "x0,y0,x1,y1,cx0,cy0,cx1,cy1", args);
        }
        circle(...args) {
            return this._glyph(glyphs_1.Circle, "x,y", args);
        }
        ellipse(...args) {
            return this._glyph(glyphs_1.Ellipse, "x,y,width,height", args);
        }
        harea(...args) {
            return this._glyph(glyphs_1.HArea, "x1,x2,y", args);
        }
        hbar(...args) {
            return this._glyph(glyphs_1.HBar, "y,height,right,left", args);
        }
        hex_tile(...args) {
            return this._glyph(glyphs_1.HexTile, "q,r", args);
        }
        image(...args) {
            return this._glyph(glyphs_1.Image, "color_mapper,image,rows,cols,x,y,dw,dh", args);
        }
        image_rgba(...args) {
            return this._glyph(glyphs_1.ImageRGBA, "image,rows,cols,x,y,dw,dh", args);
        }
        image_url(...args) {
            return this._glyph(glyphs_1.ImageURL, "url,x,y,w,h", args);
        }
        line(...args) {
            return this._glyph(glyphs_1.Line, "x,y", args);
        }
        multi_line(...args) {
            return this._glyph(glyphs_1.MultiLine, "xs,ys", args);
        }
        multi_polygons(...args) {
            return this._glyph(glyphs_1.MultiPolygons, "xs,ys", args);
        }
        oval(...args) {
            return this._glyph(glyphs_1.Oval, "x,y,width,height", args);
        }
        patch(...args) {
            return this._glyph(glyphs_1.Patch, "x,y", args);
        }
        patches(...args) {
            return this._glyph(glyphs_1.Patches, "xs,ys", args);
        }
        quad(...args) {
            return this._glyph(glyphs_1.Quad, "left,right,bottom,top", args);
        }
        quadratic(...args) {
            return this._glyph(glyphs_1.Quadratic, "x0,y0,x1,y1,cx,cy", args);
        }
        ray(...args) {
            return this._glyph(glyphs_1.Ray, "x,y,length", args);
        }
        rect(...args) {
            return this._glyph(glyphs_1.Rect, "x,y,width,height", args);
        }
        segment(...args) {
            return this._glyph(glyphs_1.Segment, "x0,y0,x1,y1", args);
        }
        spline(...args) {
            return this._glyph(glyphs_1.Spline, "x,y", args);
        }
        step(...args) {
            return this._glyph(glyphs_1.Step, "x,y,mode", args);
        }
        text(...args) {
            return this._glyph(glyphs_1.Text, "x,y,text", args);
        }
        varea(...args) {
            return this._glyph(glyphs_1.VArea, "x,y1,y2", args);
        }
        vbar(...args) {
            return this._glyph(glyphs_1.VBar, "x,width,top,bottom", args);
        }
        wedge(...args) {
            return this._glyph(glyphs_1.Wedge, "x,y,radius,start_angle,end_angle", args);
        }
        _scatter(args, marker) {
            return this._glyph(glyphs_1.Scatter, "x,y", args, marker != null ? { marker } : undefined);
        }
        scatter(...args) {
            return this._scatter(args);
        }
        asterisk(...args) {
            return this._scatter(args, "asterisk");
        }
        circle_cross(...args) {
            return this._scatter(args, "circle_cross");
        }
        circle_dot(...args) {
            return this._scatter(args, "circle_dot");
        }
        circle_x(...args) {
            return this._scatter(args, "circle_x");
        }
        circle_y(...args) {
            return this._scatter(args, "circle_y");
        }
        cross(...args) {
            return this._scatter(args, "cross");
        }
        dash(...args) {
            return this._scatter(args, "dash");
        }
        diamond(...args) {
            return this._scatter(args, "diamond");
        }
        diamond_cross(...args) {
            return this._scatter(args, "diamond_cross");
        }
        diamond_dot(...args) {
            return this._scatter(args, "diamond_dot");
        }
        dot(...args) {
            return this._scatter(args, "dot");
        }
        hex(...args) {
            return this._scatter(args, "hex");
        }
        hex_dot(...args) {
            return this._scatter(args, "hex_dot");
        }
        inverted_triangle(...args) {
            return this._scatter(args, "inverted_triangle");
        }
        plus(...args) {
            return this._scatter(args, "plus");
        }
        square(...args) {
            return this._scatter(args, "square");
        }
        square_cross(...args) {
            return this._scatter(args, "square_cross");
        }
        square_dot(...args) {
            return this._scatter(args, "square_dot");
        }
        square_pin(...args) {
            return this._scatter(args, "square_pin");
        }
        square_x(...args) {
            return this._scatter(args, "square_x");
        }
        star(...args) {
            return this._scatter(args, "star");
        }
        star_dot(...args) {
            return this._scatter(args, "star_dot");
        }
        triangle(...args) {
            return this._scatter(args, "triangle");
        }
        triangle_dot(...args) {
            return this._scatter(args, "triangle_dot");
        }
        triangle_pin(...args) {
            return this._scatter(args, "triangle_pin");
        }
        x(...args) {
            return this._scatter(args, "x");
        }
        y(...args) {
            return this._scatter(args, "y");
        }
        _pop_visuals(cls, props, prefix = "", defaults = {}, override_defaults = {}) {
            const _split_feature_trait = function (ft) {
                const fta = ft.split("_", 2);
                return fta.length == 2 ? fta : fta.concat([""]);
            };
            const _is_visual = function (ft) {
                const [feature, trait] = _split_feature_trait(ft);
                return (0, array_1.includes)(["line", "fill", "hatch", "text", "global"], feature) && trait !== "";
            };
            defaults = Object.assign({}, defaults);
            if (!hasOwnProperty.call(defaults, "text_color")) {
                defaults.text_color = "black";
            }
            if (!hasOwnProperty.call(defaults, "hatch_color")) {
                defaults.hatch_color = "black";
            }
            const trait_defaults = {};
            if (!hasOwnProperty.call(trait_defaults, "color")) {
                trait_defaults.color = _default_color;
            }
            if (!hasOwnProperty.call(trait_defaults, "alpha")) {
                trait_defaults.alpha = _default_alpha;
            }
            const result = {};
            const traits = new Set();
            for (const pname of (0, object_1.keys)(cls.prototype._props)) {
                if (_is_visual(pname)) {
                    const trait = _split_feature_trait(pname)[1];
                    if (hasOwnProperty.call(props, prefix + pname)) {
                        result[pname] = props[prefix + pname];
                        delete props[prefix + pname];
                    }
                    else if (!hasOwnProperty.call(cls.prototype._props, trait) && hasOwnProperty.call(props, prefix + trait)) {
                        result[pname] = props[prefix + trait];
                    }
                    else if (hasOwnProperty.call(override_defaults, trait)) {
                        result[pname] = override_defaults[trait];
                    }
                    else if (hasOwnProperty.call(defaults, pname)) {
                        result[pname] = defaults[pname];
                    }
                    else if (hasOwnProperty.call(trait_defaults, trait)) {
                        result[pname] = trait_defaults[trait];
                    }
                    if (!hasOwnProperty.call(cls.prototype._props, trait)) {
                        traits.add(trait);
                    }
                }
            }
            for (const name of traits) {
                delete props[prefix + name];
            }
            return result;
        }
        _find_uniq_name(data, name) {
            let i = 1;
            while (true) {
                const new_name = `${name}__${i}`;
                if (data[new_name] != null) {
                    i += 1;
                }
                else {
                    return new_name;
                }
            }
        }
        _fixup_values(cls, data, attrs) {
            for (const [name, value] of (0, object_1.entries)(attrs)) {
                const prop = cls.prototype._props[name];
                if (prop != null) {
                    if (prop.type.prototype instanceof properties_1.VectorSpec) {
                        if (value != null) {
                            if ((0, types_1.isArray)(value) || nd.is_NDArray(value)) {
                                let field;
                                if (data[name] != null) {
                                    if (data[name] !== value) {
                                        field = this._find_uniq_name(data, name);
                                        data[field] = value;
                                    }
                                    else {
                                        field = name;
                                    }
                                }
                                else {
                                    field = name;
                                    data[field] = value;
                                }
                                attrs[name] = { field };
                            }
                            else if ((0, types_1.isNumber)(value) || (0, types_1.isString)(value)) { // or Date?
                                attrs[name] = { value };
                            }
                        }
                    }
                }
            }
        }
        _glyph(cls, params_string, args, overrides) {
            const params = params_string.split(",");
            let attrs;
            if (args.length == 0) {
                attrs = {};
            }
            else if (args.length == 1) {
                attrs = Object.assign({}, args[0]);
            }
            else {
                if (args.length == params.length)
                    attrs = {};
                else
                    attrs = Object.assign({}, args[args.length - 1]);
                for (const [param, i] of (0, iterator_1.enumerate)(params)) {
                    attrs[param] = args[i];
                }
            }
            if (overrides != null) {
                attrs = Object.assign(Object.assign({}, attrs), overrides);
            }
            const source = (() => {
                const { source } = attrs;
                if (source == null)
                    return new models_1.ColumnDataSource();
                else if (source instanceof models_1.ColumnarDataSource)
                    return source;
                else
                    return new models_1.ColumnDataSource({ data: source });
            })();
            const data = (0, object_1.clone)(source.data);
            delete attrs.source;
            const view = attrs.view != null ? attrs.view : new models_1.CDSView({ source });
            delete attrs.view;
            const legend = attrs.legend;
            delete attrs.legend;
            const legend_label = attrs.legend_label;
            delete attrs.legend_label;
            const legend_field = attrs.legend_field;
            delete attrs.legend_field;
            const legend_group = attrs.legend_group;
            delete attrs.legend_group;
            if ([legend, legend_label, legend_field, legend_group].filter((arg) => arg != null).length > 1)
                throw new Error("only one of legend, legend_label, legend_field, legend_group can be specified");
            const name = attrs.name;
            delete attrs.name;
            const level = attrs.level;
            delete attrs.level;
            const visible = attrs.visible;
            delete attrs.visible;
            const x_range_name = attrs.x_range_name;
            delete attrs.x_range_name;
            const y_range_name = attrs.y_range_name;
            delete attrs.y_range_name;
            const glyph_ca = this._pop_visuals(cls, attrs);
            const nglyph_ca = this._pop_visuals(cls, attrs, "nonselection_", glyph_ca, { alpha: 0.1 });
            const sglyph_ca = this._pop_visuals(cls, attrs, "selection_", glyph_ca);
            const hglyph_ca = this._pop_visuals(cls, attrs, "hover_", glyph_ca);
            const mglyph_ca = this._pop_visuals(cls, attrs, "muted_", glyph_ca, { alpha: 0.2 });
            this._fixup_values(cls, data, glyph_ca);
            this._fixup_values(cls, data, nglyph_ca);
            this._fixup_values(cls, data, sglyph_ca);
            this._fixup_values(cls, data, hglyph_ca);
            this._fixup_values(cls, data, mglyph_ca);
            this._fixup_values(cls, data, attrs);
            source.data = data;
            const _make_glyph = (cls, attrs, extra_attrs) => {
                return new cls(Object.assign(Object.assign({}, attrs), extra_attrs));
            };
            const glyph = _make_glyph(cls, attrs, glyph_ca);
            const nglyph = !(0, object_1.is_empty)(nglyph_ca) ? _make_glyph(cls, attrs, nglyph_ca) : "auto";
            const sglyph = !(0, object_1.is_empty)(sglyph_ca) ? _make_glyph(cls, attrs, sglyph_ca) : "auto";
            const hglyph = !(0, object_1.is_empty)(hglyph_ca) ? _make_glyph(cls, attrs, hglyph_ca) : undefined;
            const mglyph = !(0, object_1.is_empty)(mglyph_ca) ? _make_glyph(cls, attrs, mglyph_ca) : "auto";
            const glyph_renderer = new models_1.GlyphRenderer({
                data_source: source,
                view,
                glyph,
                nonselection_glyph: nglyph,
                selection_glyph: sglyph,
                hover_glyph: hglyph,
                muted_glyph: mglyph,
                name,
                level,
                visible,
                x_range_name,
                y_range_name,
            });
            if (legend != null) {
                logging_1.logger.warn("Figure({legend: ...}) is deprecated and will be removed in bokeh 3.0. Use legend_label, legend_field or legend_group instead");
                const label = this._process_legend(legend, source);
                if (label != null)
                    this._update_legend(label, glyph_renderer);
            }
            if (legend_label != null)
                this._handle_legend_label(legend_label, this.legend, glyph_renderer);
            if (legend_field != null)
                this._handle_legend_field(legend_field, this.legend, glyph_renderer);
            if (legend_group != null)
                this._handle_legend_group(legend_group, this.legend, glyph_renderer);
            this.add_renderers(glyph_renderer);
            return glyph_renderer;
        }
        static _get_range(range) {
            if (range == null) {
                return new models_1.DataRange1d();
            }
            if (range instanceof models_1.Range) {
                return range;
            }
            if ((0, types_1.isArray)(range)) {
                if ((0, types_1.isArrayOf)(range, types_1.isString)) {
                    const factors = range;
                    return new models_1.FactorRange({ factors });
                }
                else {
                    const [start, end] = range;
                    return new models_1.Range1d({ start, end });
                }
            }
            throw new Error(`unable to determine proper range for: '${range}'`);
        }
        static _get_scale(range_input, axis_type) {
            if (range_input instanceof models_1.DataRange1d ||
                range_input instanceof models_1.Range1d) {
                switch (axis_type) {
                    case null:
                    case "auto":
                    case "linear":
                    case "datetime":
                    case "mercator":
                        return new models_1.LinearScale();
                    case "log":
                        return new models_1.LogScale();
                }
            }
            if (range_input instanceof models_1.FactorRange) {
                return new models_1.CategoricalScale();
            }
            throw new Error(`unable to determine proper scale for: '${range_input}'`);
        }
        _process_axis_and_grid(axis_type, axis_location, minor_ticks, axis_label, rng, dim) {
            const axis = this._get_axis(axis_type, rng, dim);
            if (axis != null) {
                if (axis instanceof models_1.LogAxis) {
                    if (dim == 0) {
                        this.x_scale = new models_1.LogScale();
                    }
                    else {
                        this.y_scale = new models_1.LogScale();
                    }
                }
                if (axis.ticker instanceof models_1.ContinuousTicker) {
                    axis.ticker.num_minor_ticks = this._get_num_minor_ticks(axis, minor_ticks);
                }
                axis.axis_label = axis_label;
                const grid = new models_1.Grid({ dimension: dim, ticker: axis.ticker });
                if (axis_location !== null) {
                    this.add_layout(axis, axis_location);
                }
                this.add_layout(grid);
            }
        }
        _get_axis(axis_type, range, dim) {
            switch (axis_type) {
                case null:
                    return null;
                case "linear":
                    return new models_1.LinearAxis();
                case "log":
                    return new models_1.LogAxis();
                case "datetime":
                    return new models_1.DatetimeAxis();
                case "mercator": {
                    const axis = new models_1.MercatorAxis();
                    const dimension = dim == 0 ? "lon" : "lat";
                    axis.ticker.dimension = dimension;
                    axis.formatter.dimension = dimension;
                    return axis;
                }
                case "auto":
                    if (range instanceof models_1.FactorRange)
                        return new models_1.CategoricalAxis();
                    else
                        return new models_1.LinearAxis(); // TODO: return DatetimeAxis (Date type)
                default:
                    throw new Error("shouldn't have happened");
            }
        }
        _get_num_minor_ticks(axis, num_minor_ticks) {
            if ((0, types_1.isNumber)(num_minor_ticks)) {
                if (num_minor_ticks <= 1) {
                    throw new Error("num_minor_ticks must be > 1");
                }
                return num_minor_ticks;
            }
            if (num_minor_ticks == null) {
                return 0;
            }
            if (num_minor_ticks === "auto") {
                return axis instanceof models_1.LogAxis ? 10 : 5;
            }
            throw new Error("shouldn't have happened");
        }
        _process_tools(tools) {
            if ((0, types_1.isString)(tools))
                tools = tools.split(/\s*,\s*/).filter((tool) => tool.length > 0);
            return tools.map((tool) => (0, types_1.isString)(tool) ? models_1.Tool.from_string(tool) : tool);
        }
        _process_legend(legend, source) {
            let legend_item_label = null;
            if (legend != null) {
                if ((0, types_1.isString)(legend)) {
                    legend_item_label = { value: legend };
                    if (source.columns() != null) {
                        if ((0, array_1.includes)(source.columns(), legend)) {
                            legend_item_label = { field: legend };
                        }
                    }
                }
                else {
                    legend_item_label = legend;
                }
            }
            return legend_item_label;
        }
        _update_legend(legend_item_label, glyph_renderer) {
            const { legend } = this;
            let added = false;
            for (const item of legend.items) {
                if (item.label != null && (0, eq_1.is_equal)(item.label, legend_item_label)) {
                    // XXX: remove this when vectorable properties are refined
                    const label = item.label;
                    if ("value" in label) {
                        item.renderers.push(glyph_renderer);
                        added = true;
                        break;
                    }
                    if ("field" in label && glyph_renderer.data_source == item.renderers[0].data_source) {
                        item.renderers.push(glyph_renderer);
                        added = true;
                        break;
                    }
                }
            }
            if (!added) {
                const new_item = new legend_item_1.LegendItem({ label: legend_item_label, renderers: [glyph_renderer] });
                legend.items.push(new_item);
            }
        }
        _handle_legend_label(value, legend, glyph_renderer) {
            const label = { value };
            const item = this._find_legend_item(label, legend);
            if (item != null)
                item.renderers.push(glyph_renderer);
            else {
                const new_item = new legend_item_1.LegendItem({ label, renderers: [glyph_renderer] });
                legend.items.push(new_item);
            }
        }
        _handle_legend_field(field, legend, glyph_renderer) {
            const label = { field };
            const item = this._find_legend_item(label, legend);
            if (item != null)
                item.renderers.push(glyph_renderer);
            else {
                const new_item = new legend_item_1.LegendItem({ label, renderers: [glyph_renderer] });
                legend.items.push(new_item);
            }
        }
        _handle_legend_group(name, legend, glyph_renderer) {
            const source = glyph_renderer.data_source;
            if (source == null)
                throw new Error("cannot use 'legend_group' on a glyph without a data source already configured");
            if (!(name in source.data))
                throw new Error("column to be grouped does not exist in glyph data source");
            const column = [...source.data[name]];
            const values = (0, array_1.uniq)(column).sort();
            for (const value of values) {
                const label = { value: `${value}` };
                const index = column.indexOf(value);
                const new_item = new legend_item_1.LegendItem({ label, renderers: [glyph_renderer], index });
                legend.items.push(new_item);
            }
        }
        _find_legend_item(label, legend) {
            const cmp = new eq_1.Comparator();
            for (const item of legend.items) {
                if (cmp.eq(item.label, label))
                    return item;
            }
            return null;
        }
    }
    exports.Figure = Figure;
    Figure.__name__ = "Plot";
    function figure(attributes) {
        return new Figure(attributes);
    }
    exports.figure = figure;
    async function show(obj, target) {
        const doc = new document_1.Document();
        for (const item of (0, types_1.isArray)(obj) ? obj : [obj])
            doc.add_root(item);
        await (0, dom_1.dom_ready)();
        let element;
        if (target == null) {
            element = document.body;
        }
        else if ((0, types_1.isString)(target)) {
            const found = document.querySelector(target);
            if (found != null && found instanceof HTMLElement)
                element = found;
            else
                throw new Error(`'${target}' selector didn't match any elements`);
        }
        else if (target instanceof HTMLElement) {
            element = target;
        }
        else if (typeof $ !== "undefined" && target instanceof $) {
            element = target[0];
        }
        else {
            throw new Error("target should be HTMLElement, string selector, $ or null");
        }
        const views = await embed.add_document_standalone(doc, element);
        return new Promise((resolve, _reject) => {
            const result = (0, types_1.isArray)(obj) ? views : views[0];
            if (doc.is_idle)
                resolve(result);
            else
                doc.idle.connect(() => resolve(result));
        });
    }
    exports.show = show;
},
434: /* api/gridplot.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    const models_1 = require(432) /* ./models */;
    const matrix_1 = require(435) /* ../core/util/matrix */;
    const assert_1 = require(11) /* ../core/util/assert */;
    function or_else(value, default_value) {
        if (value === undefined)
            return default_value;
        else
            return value;
    }
    function gridplot(children, options = {}) {
        const toolbar_location = or_else(options.toolbar_location, "above");
        const merge_tools = or_else(options.merge_tools, true);
        const sizing_mode = or_else(options.sizing_mode, null);
        const matrix = matrix_1.Matrix.from(children);
        const items = [];
        const toolbars = [];
        for (const [item, row, col] of matrix) {
            if (item == null)
                continue;
            if (item instanceof models_1.Plot) {
                if (merge_tools) {
                    toolbars.push(item.toolbar);
                    item.toolbar_location = null;
                }
                if (options.plot_width != null)
                    item.width = options.plot_width;
                if (options.plot_height != null)
                    item.height = options.plot_height;
            }
            if (options.width != null)
                item.width = options.width;
            if (options.height != null)
                item.height = options.height;
            items.push([item, row, col]);
        }
        const grid = new models_1.GridBox({ children: items, sizing_mode });
        if (!merge_tools || toolbar_location == null)
            return grid;
        const tools = [];
        for (const toolbar of toolbars) {
            tools.push(...toolbar.tools);
        }
        const toolbar = new models_1.ToolbarBox({
            toolbar: new models_1.ProxyToolbar({ toolbars, tools }),
            toolbar_location,
        });
        switch (toolbar_location) {
            case "above":
                return new models_1.Column({ children: [toolbar, grid], sizing_mode });
            case "below":
                return new models_1.Column({ children: [grid, toolbar], sizing_mode });
            case "left":
                return new models_1.Row({ children: [toolbar, grid], sizing_mode });
            case "right":
                return new models_1.Row({ children: [grid, toolbar], sizing_mode });
            default:
                (0, assert_1.unreachable)();
        }
    }
    exports.gridplot = gridplot;
},
435: /* core/util/matrix.js */ function _(require, module, exports, __esModule, __esExport) {
    __esModule();
    const array_1 = require(9) /* ./array */;
    class Matrix {
        constructor(nrows, ncols, init) {
            this.nrows = nrows;
            this.ncols = ncols;
            this._matrix = new Array(nrows);
            for (let y = 0; y < nrows; y++) {
                this._matrix[y] = new Array(ncols);
                for (let x = 0; x < ncols; x++) {
                    this._matrix[y][x] = init(y, x);
                }
            }
        }
        at(row, col) {
            return this._matrix[row][col];
        }
        *[Symbol.iterator]() {
            for (let y = 0; y < this.nrows; y++) {
                for (let x = 0; x < this.ncols; x++) {
                    const value = this._matrix[y][x];
                    yield [value, y, x];
                }
            }
        }
        *values() {
            for (const [item] of this) {
                yield item;
            }
        }
        map(fn) {
            return new Matrix(this.nrows, this.ncols, (row, col) => fn(this.at(row, col), row, col));
        }
        apply(obj) {
            const fn = Matrix.from(obj);
            const { nrows, ncols } = this;
            if (nrows == fn.nrows && ncols == fn.ncols)
                return new Matrix(nrows, ncols, (row, col) => fn.at(row, col)(this.at(row, col), row, col));
            else
                throw new Error("dimensions don't match");
        }
        to_sparse() {
            return [...this];
        }
        static from(obj, ncols) {
            if (obj instanceof Matrix) {
                return obj;
            }
            else if (ncols != null) {
                const entries = obj;
                const nrows = Math.floor(entries.length / ncols);
                return new Matrix(nrows, ncols, (row, col) => entries[row * ncols + col]);
            }
            else {
                const arrays = obj;
                const nrows = obj.length;
                const ncols = (0, array_1.min)(arrays.map((row) => row.length));
                return new Matrix(nrows, ncols, (row, col) => arrays[row][col]);
            }
        }
    }
    exports.Matrix = Matrix;
    Matrix.__name__ = "Matrix";
},
}, 427, {"api/main":427,"api/index":428,"api/linalg":429,"api/charts":430,"api/palettes":431,"api/models":432,"api/plotting":433,"api/gridplot":434,"core/util/matrix":435}, {});});
//# sourceMappingURL=bokeh-api.js.map
