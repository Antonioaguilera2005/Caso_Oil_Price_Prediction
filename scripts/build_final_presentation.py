"""Build the five-content-slide UAX deck from the selected model's saved results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import pandas as pd
from pptx import Presentation
from pptx.oxml.ns import qn
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/presentation'
ASSETS = OUT / 'assets'
BLUE, NAVY, GREY = '0060AF', '16364F', '5C6E7E'
PALE, LIGHT, AMBER, WHITE = 'EDF5FB', 'F5F7F9', '936527', 'FFFFFF'
FONT = 'Arial'
W, H = 13.333333, 7.5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def num(value, places=3):
    return f'{value:.{places}f}'.replace('-', '−').replace('.', ',')


def box(slide, x, y, w, h, fill=PALE, line=None, radius=False):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
                                   Inches(x), Inches(y), Inches(w), Inches(h))
    shape.fill.solid(); shape.fill.fore_color.rgb = RGBColor.from_string(fill)
    if line:
        shape.line.color.rgb = RGBColor.from_string(line); shape.line.width = Pt(.6)
    else:
        shape.line.fill.background()
    if radius:
        shape.adjustments[0] = .08
    return shape


def text(slide, x, y, w, h, value, size=17, color=NAVY, bold=False, align=PP_ALIGN.LEFT):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    shape.name = 'PRESENTACION_' + str(len(slide.shapes))
    frame = shape.text_frame
    frame.clear(); frame.word_wrap = True; frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = Inches(.015)
    frame.margin_top = frame.margin_bottom = Inches(.01)
    frame.vertical_anchor = MSO_ANCHOR.TOP
    for index, line in enumerate(value.split('\n')):
        p = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        p.text = line; p.alignment = align; p.line_spacing = 1.13
        p.space_after = Pt(5)
        for run in p.runs:
            run.font.name = FONT; run.font.size = Pt(size); run.font.bold = bold
            run.font.color.rgb = RGBColor.from_string(color)
            run._r.get_or_add_rPr().set('lang', 'es-ES')
    return shape


def image(slide, name, x, y, w):
    return slide.shapes.add_picture(str(ASSETS / name), Inches(x), Inches(y), width=Inches(w))


def table(slide, x, y, widths, headers, rows, *, row_height=.41, selected=None, fontsize=14):
    result = slide.shapes.add_table(len(rows)+1, len(headers), Inches(x), Inches(y),
                                   Inches(sum(widths)), Inches(row_height*(len(rows)+1)))
    result.name = 'PRESENTACION_TABLA_' + str(len(slide.shapes))
    tbl = result.table
    for c, width in zip(tbl.columns, widths):
        c.width = Inches(width)
    for i, values in enumerate([headers, *rows]):
        tbl.rows[i].height = Inches(row_height)
        for j, value in enumerate(values):
            cell = tbl.cell(i,j); cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(.08)
            cell.margin_top = cell.margin_bottom = Inches(.065)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            fill = BLUE if i == 0 else PALE if selected is not None and i-1 == selected else WHITE if i % 2 else LIGHT
            cell.fill.solid(); cell.fill.fore_color.rgb = RGBColor.from_string(fill)
            for p in cell.text_frame.paragraphs:
                p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
                p.space_after = Pt(0)
                for run in p.runs:
                    run.font.name = FONT; run.font.size = Pt(fontsize)
                    run.font.bold = i == 0 or selected is not None and i-1 == selected
                    run.font.color.rgb = RGBColor.from_string(WHITE if i == 0 else NAVY)
    return result


def heading(slide, title, section, number):
    text(slide, 8.0, .43, 4.77, .27, 'CASO 2 · REDES RECURRENTES', 11, GREY, align=PP_ALIGN.RIGHT)
    text(slide, .55, 1.08, 12.15, .53, title, 28, NAVY, True)
    box(slide, .55, 1.76, 12.2, .022, BLUE)
    text(slide, .55, 7.06, 11.0, .22, section + ' · RNN con señales · semilla 42', 9.5, GREY)
    text(slide, 12.05, 7.02, .7, .3, f'{number} / 5', 11, GREY, align=PP_ALIGN.RIGHT)


def remove_speaker_notes(presentation):
    """Remove inherited note pages and their master from the generated deck."""
    for slide in presentation.slides:
        for rel_id, relation in list(slide.part.rels.items()):
            if relation.reltype.endswith('/notesSlide'):
                slide.part.drop_rel(rel_id)
    masters = presentation._element.find(qn('p:notesMasterIdLst'))
    if masters is not None:
        presentation._element.remove(masters)
    for rel_id, relation in list(presentation.part.rels.items()):
        if relation.reltype.endswith('/notesMaster'):
            presentation.part.drop_rel(rel_id)


def clear_shape(shape):
    element = shape._element; element.getparent().remove(element)


def replace_cover(prs):
    slide = prs.slides[0]
    for shape in list(slide.shapes):
        if shape.has_text_frame and shape.shape_type == 17:
            clear_shape(shape)
    text(slide, .94, 1.35, 8.8, .35, 'CASO 2 · APRENDIZAJE AUTOMÁTICO', 16, WHITE, True)
    text(slide, .94, 2.07, 9.65, 1.65, 'Predicción del precio\ndel petróleo', 43, WHITE, True)
    text(slide, .94, 4.03, 8.9, .95, 'RNN con señales\nBrent a 5, 63 y 252 días', 24, WHITE)
    text(slide, .94, 5.25, 8.1, .65, 'Antonio Aguilera\n20 de noviembre de 2026', 16, WHITE)


def slide_method(slide, diagnostics):
    heading(slide, 'Una RNN compacta para tres horizontes', 'Datos, partición y arquitectura', 1)
    box(slide, .55, 2.02, 12.2, .49, PALE)
    text(slide, .72, 2.12, 11.85, .3,
         'Objetivo: anticipar el Brent y contrastar la previsión con mantener el último precio conocido.', 17, BLUE)
    for x, title, period, desc in [(.55, 'ENTRENAMIENTO', 'Hasta 2017', 'Pesos y escalador'),
                                  (4.68, 'VALIDACIÓN', '2018–2022', 'Selección de época'),
                                  (8.81, 'EVALUACIÓN HISTÓRICA', '2023–2024', 'Contraste con precios reales')]:
        box(slide, x, 2.79, 3.94, .99, LIGHT)
        text(slide, x+.15, 2.91, 3.6, .22, title, 11.5, GREY, True)
        text(slide, x+.15, 3.13, 3.6, .42, period, 22, BLUE, True)
        text(slide, x+.15, 3.57, 3.6, .19, desc, 11.5, GREY)
    text(slide, .55, 4.10, 5.7, .28, 'DATOS Y SEÑALES', 13, BLUE, True)
    text(slide, .55, 4.54, 5.85, 1.82,
         '• Siete variables: Brent, WTI, VIX, oro, cobre, dólar y futuro.\n'
         '• 16 señales de cambios, volatilidad, medias y base spot/futuro.\n'
         '• Escalador de entrenamiento; objetivo sin imputar.', 16.5)
    text(slide, 6.85, 4.10, 5.9, .28, 'ARQUITECTURA ELEGIDA', 13, BLUE, True)
    for x, label, detail, fill, color in [(6.85, '60 × 16', 'Entrada temporal', PALE, NAVY),
                                         (8.88, 'RNN · 32', 'Una capa tanh', BLUE, WHITE),
                                         (10.91, '3 cambios', '5 · 63 · 252 días', PALE, NAVY)]:
        box(slide, x, 4.62, 1.84, .85, fill, radius=True)
        text(slide, x+.08, 4.79, 1.68, .36, label, 19, color, True, PP_ALIGN.CENTER)
        text(slide, x+.05, 5.17, 1.74, .2, detail, 11.5, color, align=PP_ALIGN.CENTER)
    for x in [8.72,10.75]:
        box(slide, x, 4.98, .11, .06, BLUE)
    text(slide, 6.85, 5.75, 5.9, .42, 'Precio previsto = Brent conocido + cambio', 17, BLUE, True)
    text(slide, 6.85, 6.25, 5.9, .62,
         f'1.699 parámetros · LR 0,001 · batch 128\nCPU: {num(diagnostics["fit_seconds"],2)} s para este ajuste', 14, GREY)
    text(slide, .55, 6.53, 5.8, .30, 'Histórico desde 2000; entradas completas desde 2012.', 13, GREY)


def slide_training(slide, diagnostics):
    heading(slide, 'Elegir la época también controla el sobreajuste', 'Entrenamiento y generalización', 2)
    text(slide, .58, 2.02, 5.7, .27, 'CURVA DE APRENDIZAJE', 13, BLUE, True)
    text(slide, 6.95, 2.02, 5.8, .27, 'VALIDACIÓN FRENTE A TEST HISTÓRICO', 13, BLUE, True)
    image(slide, 'final_training_curve.png', .52, 2.40, 5.65)
    image(slide, 'final_validation_test.png', 6.89, 2.40, 5.85)
    box(slide, .55, 5.70, 5.94, 1.18, PALE)
    text(slide, .72, 5.83, 5.6, .3, 'Checkpoint: época 17 de 27 ejecutadas', 18, BLUE, True)
    text(slide, .72, 6.24, 5.6, .53, 'Early stopping, paciencia 10. El score empeoró un 2,7 % tras la época elegida.', 14)
    box(slide, 6.81, 5.70, 5.94, 1.18, LIGHT)
    text(slide, 6.98, 5.83, 5.6, .3, 'Exploración previa: 112 intentos de Optuna', 18, NAVY, True)
    text(slide, 6.98, 6.24, 5.6, .53, 'RNN, LSTM y GRU; 62 completos y 50 podados. Más complejidad no aseguró una ventaja.', 14)


def slide_curves(slide):
    heading(slide, 'Las curvas se separan al ampliar el horizonte', 'Precio real frente a predicho · 2023–2024', 3)
    image(slide, 'final_test_curves.png', .57, 2.03, 12.15)
    captions=[('5 días', 'Mayor ajuste a las oscilaciones'), ('63 días', 'Desviaciones más persistentes'), ('252 días', 'Poca anticipación de los giros')]
    for x,(label,desc) in zip([.70,4.86,9.02],captions):
        text(slide, x, 6.12, 3.75, .35, label, 18, BLUE, True)
        text(slide, x, 6.49, 3.75, .26, desc, 14)


def slide_metrics(slide, all_metrics, metrics):
    heading(slide, 'Qué mejora y qué no mejora frente al ingenuo', 'Comparación y métricas guardadas', 4)
    text(slide, .55, 2.04, 5.75, .4, 'Configuraciones · RMSE / ingenuo spot', 17, BLUE, True)
    order=[('RNN','signals'),('RNN','levels'),('LSTM','signals'),('LSTM','levels'),('GRU','signals'),('GRU','levels')]
    rows=[]
    for architecture,variant in order:
        subset=all_metrics.loc[(all_metrics.architecture==architecture)&(all_metrics.variant==variant)&(all_metrics.group=='all')].set_index('horizon')
        rows.append([architecture+' · '+('señales' if variant=='signals' else 'niveles'),*[num(subset.loc[h,'relative_rmse']) for h in [5,63,252]]])
    table(slide,.55,2.62,[2.22,1.09,1.09,1.09],['Red','5 días','63 días','252 días'],rows,row_height=.43,selected=0,fontsize=13.5)
    text(slide,.55,5.81,5.55,.60,'Menos de 1 = menor RMSE.\nMisma partición y semilla; selección de configuraciones.',13,GREY)
    text(slide,6.65,2.04,6.05,.4,'RNN con señales · evaluación 2023–2024',17,BLUE,True)
    metric_rows=[]
    for row in metrics.itertuples():
        metric_rows.append([row.horizon,row.n,num(row.mae),num(row.mse,2),num(row.rmse),num(row.r2)])
    table(slide,6.65,2.62,[.62,.61,1.13,1.25,1.13,1.12],['Días','N','MAE','MSE','RMSE','R²'],metric_rows,row_height=.55,fontsize=13.5)
    text(slide,6.65,4.97,6.04,.37,'MAE y RMSE: USD/barril · MSE: (USD/barril)².',12.5,GREY)
    text(slide,6.65,5.49,6.04,.75,'Ratio frente a futuros conocidos:\n1,080 · 1,073 · 1,081 (5 · 63 · 252 días)',15,AMBER,True)
    box(slide,.55,6.53,12.2,.39,PALE)
    text(slide,.70,6.59,11.9,.31,'5 días: 11,4 % menos RMSE que el spot. A 63 y 252 días, empate práctico.',16,BLUE,True)


def slide_errors_app(slide):
    heading(slide, 'Conocer los errores forma parte de la solución', 'Errores, límites y demostración de la app', 5)
    text(slide,.55,2.04,6.1,.28,'ERROR RELATIVO SEGÚN VOLATILIDAD EN EL ORIGEN',12.5,BLUE,True)
    text(slide,7.1,2.04,5.65,.28,'EL MISMO MODELO, TRES HORIZONTES',12.5,BLUE,True)
    image(slide,'final_error_regimes.png',.51,2.43,6.13)
    image(slide,'app_demo.png',7.1,2.47,5.65)
    text(slide,.55,5.94,6.1,.66,'Alta volatilidad: N = 26 / 11 / 9.\nComparación descriptiva; grupos pequeños y solapados.',14,GREY)
    text(slide,7.1,5.69,5.65,.96,'• CSV para incorporar datos sin reentrenar.\n• Rangos empíricos nominales del 80 %.\n• Demostración con instantánea del 13/11/2026.',14)
    box(slide,.55,6.68,12.2,.30,PALE)
    text(slide,.70,6.70,11.9,.27,'Valor: apoyo al análisis, con una ventaja medible a cinco días y límites visibles.',16,BLUE,True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template',type=Path,default=OUT/'plantilla_UAX_original.pptx',
                        help='UAX template path; defaults to the copy included in docs/presentation.')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    template=OUT/'plantilla_UAX_original.pptx'
    if args.template.resolve() != template.resolve():
        shutil.copy2(args.template,template)
    delivery=json.loads((ROOT/'config/delivery_model.json').read_text(encoding='utf-8'))
    source=ROOT/delivery['evaluation']
    all_metrics=pd.read_csv(source/'metrics.csv')
    metrics=all_metrics.loc[(all_metrics.architecture=='RNN')&(all_metrics.variant=='signals')&(all_metrics.seed==42)&(all_metrics.group=='all')].sort_values('horizon')
    diagnostics=json.loads((ROOT/delivery['checkpoint']).with_name('diagnostics.json').read_text())
    before={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/delivery['checkpoint'],source/'metrics.csv',source/'predictions.csv']}
    prs=Presentation(template)
    # Keep the original cover/closing artwork and use its UAX content layout.
    layout=prs.slides[1].slide_layout
    # Add before removing blanks so python-pptx allocates unique slide part names.
    slides=[prs.slides.add_slide(layout) for _ in range(5)]
    for index in [2,1]:
        slide_id=prs.slides._sldIdLst[index]
        prs.part.drop_rel(slide_id.rId);prs.slides._sldIdLst.remove(slide_id)
    replace_cover(prs)
    slide_method(slides[0],diagnostics)
    slide_training(slides[1],diagnostics)
    slide_curves(slides[2])
    slide_metrics(slides[3],all_metrics,metrics)
    slide_errors_app(slides[4])
    closing=prs.slides[1]
    for shape in list(closing.shapes):
        if shape.has_text_frame and 'Thank' in shape.text:
            clear_shape(shape)
    text(closing,2.2,2.65,8.9,1.75,'Gracias\n¿Preguntas?',44,WHITE,True,PP_ALIGN.CENTER)
    closing_id=prs.slides._sldIdLst[1]
    prs.slides._sldIdLst.remove(closing_id);prs.slides._sldIdLst.append(closing_id)
    prs.core_properties.title='Predicción del precio del petróleo · RNN con señales'
    prs.core_properties.subject='Modelo, validación, métricas, gráficas y análisis de errores'
    prs.core_properties.author='Antonio Aguilera'
    prs.core_properties.keywords='UAX, Brent, RNN, series temporales, evaluación histórica'
    output=OUT/'Prediccion_Petroleo_RNN_UAX.pptx'
    remove_speaker_notes(prs)
    prs.save(output)
    with zipfile.ZipFile(output) as archive:
        assert len(archive.namelist()) == len(set(archive.namelist())), 'Duplicate presentation ZIP parts'
    reopened=Presentation(output)
    assert len(reopened.slides)==7
    assert not any(s.has_notes_slide for s in reopened.slides)
    assert not any('xxxxxxxx' in sh.text or 'yyyyyyyy' in sh.text for s in reopened.slides for sh in s.shapes if sh.has_text_frame)
    outside=[]
    for i,s in enumerate(reopened.slides,1):
        for sh in s.shapes:
            if sh.name.startswith('PRESENTACION_') and (sh.left<0 or sh.top<0 or sh.left+sh.width>reopened.slide_width+100 or sh.top+sh.height>reopened.slide_height+100):
                outside.append([i,sh.name])
    assert not outside,outside
    assert before=={p:sha(ROOT/p) for p in before}
    report={'pptx':output.name,'template':str(args.template),'template_sha256':sha(args.template),'slides':7,'content_slides':5,
            'model_id':delivery['id'],'source_files':before,'source_files_unchanged':True,'speaker_notes':False,
            'editable_text_and_tables':True,'generated_shapes_within_slide':True,'figures_manifest':'assets/final_figure_manifest.json',
            'rendered_with_powerpoint':False}
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'presentation':str(output),'slides':7,'content_slides':5,'source_files_unchanged':True},ensure_ascii=False))


if __name__=='__main__':
    main()
