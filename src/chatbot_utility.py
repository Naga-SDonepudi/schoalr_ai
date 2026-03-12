import os


working_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(working_dir)

def get_chapter_list(selected_subject):
        subject_name = selected_subject.lower()
        chapters_dir = os.path.join(parent_dir, "data" ,subject_name)

        if not os.path.exists(chapters_dir):
            print(f"Directory Not Found: {chapters_dir}")
            return []

        chapters_list = [f[:-4] for f in os.listdir(chapters_dir) if f.endswith('.pdf')]     #Excludes .pdf extension when answering

        try:
            chapters_list.sort(key=lambda x: int(x.split('.')[0]))
        except ValueError:
            chapters_list.sort()

        return chapters_list



# chapters_list = get_chapter_list("ML_and_DL")
# print(chapters_list)
