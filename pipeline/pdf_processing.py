import os
import re
import fitz
import numpy as np
import cv2
from datetime import datetime


from config import model_detect, model_classify

def iou(box1, box2):
    x1, y1, x2, y2 = box1
    X1, Y1, X2, Y2 = box2
    inter_area = max(0, min(x2,X2)-max(x1,X1)) * max(0, min(y2,Y2)-max(y1,Y1))
    box1_area = (x2-x1)*(y2-y1)
    box2_area = (X2-X1)*(Y2-Y1)
    union_area = box1_area + box2_area - inter_area
    return inter_area/union_area if union_area>0 else 0

def is_contained(inner, outer, tol=0.95):
    x1, y1, x2, y2 = inner
    X1, Y1, X2, Y2 = outer
    ix1, iy1, ix2, iy2 = max(x1,X1), max(y1,Y1), min(x2,X2), min(y2,Y2)
    inter_area = max(0, ix2-ix1)*max(0, iy2-iy1)
    inner_area = (x2-x1)*(y2-y1)
    return inner_area>0 and (inter_area/inner_area)>=tol

def filter_duplicates(crops, iou_thresh=0.7):
    keep=[]
    crops=sorted(crops,key=lambda x:(x[2][2]-x[2][0])*(x[2][3]-x[2][1]), reverse=True)
    for crop_img, conf, box in crops:
        if not any(iou(box, kept_box)>iou_thresh or is_contained(box, kept_box) for _,_,kept_box in keep):
            keep.append((crop_img, conf, box))
    return keep

def extract_clean_text_with_inline_placeholders(pdf_path, page_placeholders, min_words=5):
    doc = fitz.open(pdf_path)
    all_pages=[]
    for page_num, page in enumerate(doc, start=1):
        page_lines=[f"--- Page {page_num} ---"]
        placeholders=page_placeholders.get(page_num,[])
        used_placeholders=set()
        words=page.get_text("words")
        lines={}
        for w in words:
            x0,y0,x1,y1,word,block_no,line_no,_=w
            key=(block_no,line_no)
            if key not in lines: lines[key]={"y0":y0,"y1":y1,"words":[]}
            lines[key]["words"].append(word)
        text_blocks=[]
        for _,data in lines.items():
            txt=" ".join(data["words"]).strip()
            if txt and len(txt.split())>=min_words and not re.match(r'^[\d\.\%\$\s,-]+$', txt):
                text_blocks.append({"y0":data["y0"],"y1":data["y1"],"text":txt})
        text_blocks=sorted(text_blocks,key=lambda x:x["y0"])
        for t in text_blocks:
            for i, ph in enumerate(placeholders):
                if i in used_placeholders: continue
                px0, py0, px1, py1=ph["bbox"]
                if not (t["y1"]<py0 or t["y0"]>py1):
                    page_lines.append(ph["placeholder"])
                    used_placeholders.add(i)
            page_lines.append(t["text"])
        for i, ph in enumerate(placeholders):
            if i not in used_placeholders:
                page_lines.append(ph["placeholder"])
        all_pages.append("\n".join(page_lines))
    return "\n\n".join(all_pages)

def extract_report_date_from_filename(filename: str):
    """
    filename: e.g., 'TrendForce_ServerDRAM_Aug2022.pdf'
    Returns: str in 'YYYY-MM' format or None if not found
    """
    base_name = os.path.splitext(filename)[0]
    match = re.search(
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-zA-Z]*\.?[- ]?([0-9]{2,4})",
        base_name,
        re.IGNORECASE
    )
    if match:
        month_str, year_str = match.groups()
        if len(year_str) == 2:
            year_str = "20" + year_str
        month_num = datetime.strptime(month_str[:3], "%b").month
        return f"{year_str}-{month_num:02d}"
    return None

def process_pdfs(pdf_input_folder):
    all_placeholders, all_text_json, all_table_chart_nodes = {}, [], []
    FIGURE_CLASS, FIGURE_CAPTION_CLASS, TABLE_CLASS, TABLE_CAPTION_CLASS = 3,4,5,6
    CONF_THRESHOLD=0

    for pdf_file in os.listdir(pdf_input_folder):
        if not pdf_file.lower().endswith(".pdf"): continue
        pdf_path=os.path.join(pdf_input_folder,pdf_file)
        pdf_name=os.path.splitext(pdf_file)[0]
        clean_pdf_name=re.sub(r'[^A-Za-z0-9]+','',pdf_name)
        report_date = extract_report_date_from_filename(pdf_file)
        doc=fitz.open(pdf_path)
        page_placeholders={}

        # --- Extract charts & tables ---
        for page_idx, page in enumerate(doc):
            pix=page.get_pixmap(matrix=fitz.Matrix(2,2))
            img=np.frombuffer(pix.samples,dtype=np.uint8).reshape(pix.height,pix.width,pix.n)
            img=cv2.cvtColor(img, cv2.COLOR_RGBA2BGR) if pix.n==4 else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            results=model_detect(img, save=False)
            for r in results:
                boxes=r.boxes
                figures, fig_caps, tables, tab_caps=[],[],[],[]
                for box in boxes:
                    conf=float(box.conf[0]); cls=int(box.cls[0])
                    xyxy=box.xyxy[0].cpu().numpy().astype(int)
                    if conf>=CONF_THRESHOLD:
                        if cls==FIGURE_CLASS: figures.append((xyxy, conf))
                        elif cls==FIGURE_CAPTION_CLASS: fig_caps.append((xyxy, conf))
                        elif cls==TABLE_CLASS: tables.append((xyxy, conf))
                        elif cls==TABLE_CAPTION_CLASS: tab_caps.append((xyxy, conf))

                # Charts
                chart_crops=[]
                for fig_box,fconf in figures:
                    merged_box=fig_box.copy()
                    for cap_box,_ in fig_caps:
                        if abs(cap_box[1]-fig_box[3])<150 or abs(fig_box[1]-cap_box[3])<150:
                            merged_box=[min(merged_box[0],cap_box[0]),min(merged_box[1],cap_box[1]),
                                        max(merged_box[2],cap_box[2]),max(merged_box[3],cap_box[3])]
                    x1,y1,x2,y2=merged_box; w=x2-x1; pad_w=int(0.05*w); x1=max(0,x1-pad_w); x2=min(img.shape[1],x2+pad_w)
                    crop=img[y1:y2, x1:x2]
                    class_result=model_classify(crop, save=False)[0]
                    pred_cls=int(class_result.probs.top1); pred_conf=float(class_result.probs.top1conf)
                    if pred_cls==0 and pred_conf>=0.4: chart_crops.append((crop,fconf,merged_box))
                chart_crops=filter_duplicates(chart_crops)
                for idx,(crop,_,box) in enumerate(chart_crops):
                    placeholder=f"[{clean_pdf_name}_Chart{idx+1}_Page{page_idx+1}]"
                    page_placeholders.setdefault(page_idx+1,[]).append({"placeholder":placeholder,"bbox":[int(c) for c in box]})
                    node={"img":crop,"placeholder":placeholder,"node-type":"chart","page_num":page_idx+1}
                    all_table_chart_nodes.append(node)

                # Tables
                table_crops=[]
                for tab_box,tconf in tables:
                    merged_box=tab_box.copy()
                    for cap_box,_ in tab_caps:
                        if abs(cap_box[1]-tab_box[3])<150 or abs(tab_box[1]-cap_box[3])<150:
                            merged_box=[min(merged_box[0],cap_box[0]),min(merged_box[1],cap_box[1]),
                                        max(merged_box[2],cap_box[2]),max(merged_box[3],cap_box[3])]
                    x1,y1,x2,y2=merged_box; w=x2-x1; h=y2-y1; pad_w=int(0.05*w); pad_h=int(0.05*h)
                    x1=max(0,x1-pad_w); x2=min(img.shape[1],x2+pad_w); y1=max(0,y1-pad_h)
                    crop=img[y1:y2, x1:x2]
                    table_crops.append((crop,tconf,[x1,y1,x2,y2]))
                table_crops=filter_duplicates(table_crops)
                for idx,(crop,_,box) in enumerate(table_crops):
                    placeholder=f"[{clean_pdf_name}_Table{idx+1}_Page{page_idx+1}]"
                    page_placeholders.setdefault(page_idx+1,[]).append({"placeholder":placeholder,"bbox":[int(c) for c in box]})
                    node={"img":crop,"placeholder":placeholder,"node-type":"table","page_num":page_idx+1}
                    all_table_chart_nodes.append(node)

        all_placeholders[clean_pdf_name]=page_placeholders

        # Extract text
        pdf_text_with_placeholders=extract_clean_text_with_inline_placeholders(pdf_path,page_placeholders)
        pages_split=pdf_text_with_placeholders.split("--- Page ")
        for page_content in pages_split:
            if not page_content.strip(): continue
            header,*body_lines=page_content.split("\n")
            try: page_num=int(header.strip("- ").strip())
            except: page_num=None
            content="\n".join([line for line in body_lines if line.strip()])
            placeholders=re.findall(r"\[.*?_Table\d+_Page\d+\]|\[.*?_Chart\d+_Page\d+\]", content)
            content=re.sub(r"^\[.*?_Table\d+_Page\d+\]$|^\[.*?_Chart\d+_Page\d+\]$","",content,flags=re.MULTILINE)
            content="\n".join([l for l in content.splitlines() if l.strip()])
            node={"id":f"{clean_pdf_name}_page{page_num}","report_date":report_date,"page_num":page_num,"node-type":"page_text","placeholder":placeholders,"text":content}
            all_text_json.append(node)

    return all_placeholders, all_text_json, all_table_chart_nodes