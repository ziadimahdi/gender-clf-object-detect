"""Main module for the streamlit app"""
import streamlit as st

import awesome_streamlit as ast
import apps.app1
import apps.app2


ast.core.services.other.set_logging_format()

PAGES = {
    "Gender Classification": apps.app1,
    "Object Detection": apps.app2,

}


def main():
    """Main function of the App"""
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", list(PAGES.keys()))

    page = PAGES[selection]

    with st.spinner(f"Loading {selection} ..."):
        ast.shared.components.write_page(page)

    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app is maintained by Marc Skov Madsen. You can learn more about this app at
        [GitHub](https://github.com/ziadimahdi/Gender-Classification-Object-Detection.git).
"""
    )


if __name__ == "__main__":
    main()
